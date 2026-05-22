#!/usr/bin/env python
"""Compute hidden-state embeddings with the target LLM (e.g., Llama-3.1-8B).

This is intended for DATE-LM style baselines (Rep-Sim / RDS+) and for running BipCov
with LLM hidden-state embeddings.

Outputs (under --out_dir):
  - train_emb.npy (optional)
  - ref_emb.npy (optional)
  - meta.json
  - progress.json (for resumable train embeddings)

Notes
- Train JSONL is expected to contain a `messages` field (OpenAI chat format).
- For train data we mirror DATE-LM `only_first_two=True` behavior by default.
- For reference data we support generating prompts via DATE-LM's
  `minimal_multitask.data.DATASETS` (e.g., mmlu/gsm8k/bbh and *_shots), which requires
  running from DATE-LM root (so `data/` is present).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm


def _messages_first_two(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    if messages[0].get("role") == "system":
        return messages[1:3]
    return messages[:2]


def _format_messages_tulu(messages: list[dict], eos_token: str) -> str:
    out = ""
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "system":
            out += "<|system|>\n" + content + "\n"
        elif role == "user":
            out += "<|user|>\n" + content + "\n"
        elif role == "assistant":
            out += "<|assistant|>\n" + content + eos_token + "\n"
        else:
            raise ValueError(f"Unsupported role: {role}")
    return out.strip()


def _iter_jsonl_messages(path: Path, start: int = 0, max_samples: int = -1) -> Iterator[list[dict]]:
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if max_samples > 0 and (i - start) >= max_samples:
                break
            obj = json.loads(line)
            msgs = obj.get("messages")
            if not isinstance(msgs, list):
                raise ValueError("Expected each JSONL row to have a `messages` list")
            yield msgs


def _count_lines(path: Path) -> int:
    with path.open("r") as f:
        return sum(1 for _ in f)


def _load_progress(progress_path: Path) -> dict:
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text())
    except Exception:
        return {}


def _save_progress(progress_path: Path, progress: dict) -> None:
    tmp = progress_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(progress, indent=2))
    tmp.replace(progress_path)


def _to_numpy_dtype(name: str) -> np.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return np.float16
    if name in {"fp32", "float32"}:
        return np.float32
    raise ValueError(f"Unsupported --out_dtype: {name}")


def _compute_emb(
    model,
    input_ids,
    attention_mask,
    pooling: str,
    normalize: bool = True,
):
    import torch

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        if pooling == "last_token":
            seq_lens = attention_mask.sum(dim=1) - 1
            batch = last_hidden.size(0)
            reps = last_hidden[torch.arange(batch, device=last_hidden.device), seq_lens]
        elif pooling == "mean":
            # Masked mean over non-pad tokens (works for left/right padding).
            mask = attention_mask.to(dtype=last_hidden.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            reps = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif pooling == "weighted_mean":
            # Paper-faithful RDS+ weighted mean pooling over tokens:
            # w_i = i / sum_{i=1}^L i, where i is token index (1..L) and L is non-pad length.
            # Implemented via per-token rank from attention_mask.cumsum, so it works for both
            # left- and right-padding.
            pos = attention_mask.cumsum(dim=1) * attention_mask  # [B, T] with 0 for pad, 1..L for tokens
            pos = pos.to(dtype=last_hidden.dtype)
            denom = pos.sum(dim=1, keepdim=True).clamp_min(1)  # sum_{i=1}^L i
            weights = pos / denom  # [B, T]
            reps = (last_hidden * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        if normalize:
            reps = torch.nn.functional.normalize(reps, dim=-1)
        return reps


def _batched(iterable: Iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datelm_root", type=str, required=True)

    p.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dtype", type=str, default="float16")
    p.add_argument(
        "--pooling",
        type=str,
        default="last_token",
        choices=["last_token", "mean", "weighted_mean"],
        help="Embedding pooling: last_token (Rep-Sim), weighted_mean (RDS+), or mean.",
    )

    # Train
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument(
        "--messages_only_first_two",
        action="store_true",
        help="Mirror DATE-LM finetune preprocessing (only_first_two=True).",
    )
    p.add_argument("--no_train", action="store_true")

    # Ref
    p.add_argument(
        "--ref_dataset_name",
        type=str,
        default=None,
        choices=["mmlu", "gsm8k", "bbh", "mmlu_shots", "gsm8k_shots", "bbh_shots"],
        help="Generate ref prompts from DATE-LM minimal_multitask.data.DATASETS.",
    )
    p.add_argument("--ref_num_samples", type=int, default=100)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--no_ref", action="store_true")

    # Normalization (DATE-LM uses raw dot product, not cosine similarity)
    p.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip L2 normalization (use dot product like DATE-LM original).",
    )

    # Output
    p.add_argument("--out_dir", type=str, required=True)

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datelm_root = Path(args.datelm_root)
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))

    # Must run from DATE-LM root so that DATASETS can find `data/`
    os.chdir(str(datelm_root))

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    out_dtype = _to_numpy_dtype(args.out_dtype)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading base model (AutoModel): {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    do_normalize = not args.no_normalize
    meta: dict = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "out_dtype": str(out_dtype),
        "pooling": str(args.pooling),
        "messages_only_first_two": bool(args.messages_only_first_two),
        "normalize": do_normalize,
        "train": None,
        "ref": None,
    }

    progress_path = out_dir / "progress.json"
    progress = _load_progress(progress_path)

    # Train embeddings
    if not args.no_train:
        if not args.train_jsonl:
            raise ValueError("Provide --train_jsonl (or pass --no_train)")

        train_jsonl = Path(args.train_jsonl)
        if not train_jsonl.exists():
            raise FileNotFoundError(str(train_jsonl))

        n_total = _count_lines(train_jsonl)
        if args.train_max_samples > 0:
            n_total = min(n_total, int(args.train_max_samples))

        out_path = out_dir / "train_emb.npy"
        if out_path.exists() and progress.get("train_complete") is True:
            print(f"Train embeddings already complete: {out_path}")
        else:
            start_idx = int(progress.get("train_next_idx", 0))
            if start_idx < 0 or start_idx > n_total:
                start_idx = 0

            mode = "r+" if out_path.exists() else "w+"
            if not out_path.exists():
                print(f"Allocating memmap {out_path} shape=({n_total}, {model.config.hidden_size}) dtype={out_dtype}")
            train_mm = open_memmap(
                out_path,
                mode=mode,
                dtype=out_dtype,
                shape=(n_total, int(model.config.hidden_size)),
            )

            print(f"Computing train embeddings: total={n_total} start={start_idx}")
            iterator = _iter_jsonl_messages(train_jsonl, start=start_idx, max_samples=args.train_max_samples)

            write_idx = start_idx
            for batch_msgs in tqdm(_batched(iterator, args.batch_size), desc="Train emb"):
                texts = []
                for msgs in batch_msgs:
                    if args.messages_only_first_two:
                        msgs = _messages_first_two(msgs)
                    texts.append(_format_messages_tulu(msgs, eos_token=tokenizer.eos_token))

                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                reps = _compute_emb(
                    model,
                    input_ids,
                    attention_mask,
                    pooling=str(args.pooling),
                    normalize=do_normalize,
                )
                reps_np = reps.detach().to(torch.float32).cpu().numpy()
                if out_dtype == np.float16:
                    reps_np = reps_np.astype(np.float16, copy=False)
                else:
                    reps_np = reps_np.astype(np.float32, copy=False)

                end = write_idx + reps_np.shape[0]
                train_mm[write_idx:end] = reps_np
                write_idx = end

                if write_idx % 1024 == 0:
                    progress["train_next_idx"] = write_idx
                    _save_progress(progress_path, progress)

            progress["train_next_idx"] = write_idx
            progress["train_complete"] = True
            _save_progress(progress_path, progress)
            print(f"Saved train_emb -> {out_path} (rows={write_idx})")

        meta["train"] = {
            "train_jsonl": str(train_jsonl),
            "rows": int(n_total),
            "out": str(out_path),
        }

    # Ref embeddings
    if not args.no_ref:
        if not args.ref_dataset_name:
            raise ValueError("Provide --ref_dataset_name (or pass --no_ref)")

        ref_out = out_dir / "ref_emb.npy"
        if ref_out.exists():
            print(f"Ref embeddings exist: {ref_out}")
        else:
            import sys

            sys.path.insert(0, str(datelm_root))
            from minimal_multitask.data import DATASETS
            from transformers import DataCollatorForSeq2Seq
            from torch.utils.data import DataLoader

            ref_name = str(args.ref_dataset_name)
            val_ds = DATASETS[ref_name](tokenizer).get_all_test_prompts(
                num_samples=int(args.ref_num_samples),
                seed=int(args.ref_seed),
                prompt_only=False,
                response_only=False,
            )

            collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
            loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collator)

            reps_all = []
            for batch in tqdm(loader, desc=f"Ref emb ({ref_name})"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                reps = _compute_emb(
                    model,
                    input_ids,
                    attention_mask,
                    pooling=str(args.pooling),
                    normalize=do_normalize,
                )
                reps_all.append(reps.detach().to(torch.float32).cpu().numpy())

            ref = np.concatenate(reps_all, axis=0).astype(np.float32)
            np.save(ref_out, ref)
            print(f"Saved ref_emb -> {ref_out} shape={ref.shape}")

        meta["ref"] = {
            "ref_dataset_name": args.ref_dataset_name,
            "ref_num_samples": int(args.ref_num_samples),
            "ref_seed": int(args.ref_seed),
            "out": str(ref_out),
        }

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
