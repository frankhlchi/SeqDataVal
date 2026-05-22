#!/usr/bin/env python
"""Compute text embeddings with a Hugging Face model (AutoModel + pooling).

This is intended for BipCov "embedding backend" ablations while keeping the
DATE-LM train/eval pipeline unchanged (DATE-LM/train/finetune.py +
minimal_multitask official eval).

Outputs (under --out_dir):
  - train_emb.npy (optional)
  - ref_emb.npy (optional)
  - meta.json
  - progress.json (for resumable train embeddings)

Notes
- Train JSONL is expected to contain a `messages` field (OpenAI chat format).
- For train data we mirror DATE-LM finetune preprocessing (`only_first_two=True`) via
  --messages_only_first_two.
- For reference data we generate prompt+label prompts via DATE-LM's
  minimal_multitask.data.DATASETS using a base tokenizer (default: Llama-3.1-8B),
  decode to text, then embed with the target embedding model.
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


def _format_messages(messages: list[dict], mode: str, eos_token: str) -> str:
    if mode == "plain":
        lines = []
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in {"system", "user", "assistant"}:
                lines.append(f"{role.capitalize()}: {content}")
            else:
                raise ValueError(f"Unsupported role: {role}")
        return "\n".join(lines).strip()

    if mode == "tulu_chat":
        out = ""
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role == "system":
                out += "<|system|>\n" + content + "\n"
            elif role == "user":
                out += "<|user|>\n" + content + "\n"
            elif role == "assistant":
                out += "<|assistant|>\n" + content + (eos_token or "") + "\n"
            else:
                raise ValueError(f"Unsupported role: {role}")
        return out.strip()

    raise ValueError(f"Unsupported --format: {mode}")


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


def _to_torch_dtype(name: str):
    import torch

    name = name.lower()
    if name in {"auto"}:
        return None
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --model_dtype: {name}")


def _infer_hidden_size(model) -> int:
    cfg = getattr(model, "config", None)
    for key in ("hidden_size", "dim", "n_embd"):
        val = getattr(cfg, key, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            return int(val)
    raise ValueError("Could not infer hidden size from model.config")


def _compute_emb(model, input_ids, attention_mask, pooling: str, normalize: bool):
    import torch

    with torch.inference_mode():
        # Some embedding backends (e.g., NV-Embed-v2) support a separate `pool_mask`
        # and/or return a plain dict instead of a ModelOutput.
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, pool_mask=attention_mask, return_dict=True)
        except TypeError:
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        if isinstance(out, dict):
            if "sentence_embeddings" in out:
                reps = out["sentence_embeddings"]
                if reps.ndim == 3 and reps.shape[1] == 1:
                    reps = reps[:, 0, :]
                if normalize:
                    reps = torch.nn.functional.normalize(reps, dim=-1)
                return reps

            if "last_hidden_state" in out:
                last_hidden = out["last_hidden_state"]
            elif "hidden_states" in out:
                hidden_states = out["hidden_states"]
                if isinstance(hidden_states, (tuple, list)) and hidden_states:
                    last_hidden = hidden_states[-1]
                else:
                    last_hidden = hidden_states
            else:
                raise ValueError(f"Unsupported model output dict keys: {sorted(out.keys())}")
        else:
            last_hidden = getattr(out, "last_hidden_state", None)
            if last_hidden is None:
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    last_hidden = out[0]
                else:
                    raise ValueError(f"Unsupported model output type: {type(out)}")

        if pooling == "last_token":
            seq_lens = attention_mask.sum(dim=1) - 1
            batch = last_hidden.size(0)
            reps = last_hidden[torch.arange(batch, device=last_hidden.device), seq_lens]
        elif pooling == "cls":
            reps = last_hidden[:, 0, :]
        elif pooling == "mean":
            mask = attention_mask.to(dtype=last_hidden.dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            reps = (last_hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif pooling == "weighted_mean":
            pos = attention_mask.cumsum(dim=1) * attention_mask
            pos = pos.to(dtype=last_hidden.dtype)
            denom = pos.sum(dim=1, keepdim=True).clamp_min(1)
            weights = pos / denom
            reps = (last_hidden * weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

        if normalize:
            reps = torch.nn.functional.normalize(reps, dim=-1)
        return reps


def _batched(iterable: Iterable[str], batch_size: int):
    batch: list[str] = []
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

    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--model_dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--out_dtype", type=str, default="float16")
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "weighted_mean", "last_token", "cls"])
    p.add_argument("--format", type=str, default="tulu_chat", choices=["tulu_chat", "plain"])
    p.add_argument("--doc_prefix", type=str, default="")
    p.add_argument("--query_prefix", type=str, default="")
    p.add_argument("--base_tokenizer", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--messages_only_first_two", action="store_true")
    p.add_argument("--no_normalize", action="store_true")

    # Train
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument("--no_train", action="store_true")

    # Ref
    p.add_argument(
        "--ref_dataset_name",
        type=str,
        default=None,
        choices=["mmlu", "gsm8k", "bbh", "mmlu_shots", "gsm8k_shots", "bbh_shots"],
    )
    p.add_argument("--ref_num_samples", type=int, default=100)
    p.add_argument("--ref_seed", type=int, default=42)
    p.add_argument("--no_ref", action="store_true")

    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    datelm_root = Path(args.datelm_root).expanduser().resolve()
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))
    os.chdir(str(datelm_root))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device(args.device)
    torch_dtype = _to_torch_dtype(str(args.model_dtype))
    out_dtype = _to_numpy_dtype(str(args.out_dtype))
    normalize = not bool(args.no_normalize)

    print(f"Loading embedding tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id

    print(f"Loading embedding model (AutoModel): {args.model_name}")
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=bool(args.trust_remote_code),
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    hidden_size = _infer_hidden_size(model)

    meta: dict = {
        "model_name": args.model_name,
        "trust_remote_code": bool(args.trust_remote_code),
        "model_dtype": str(args.model_dtype),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "out_dtype": str(out_dtype),
        "pooling": str(args.pooling),
        "format": str(args.format),
        "doc_prefix": str(args.doc_prefix),
        "query_prefix": str(args.query_prefix),
        "base_tokenizer": str(args.base_tokenizer),
        "messages_only_first_two": bool(args.messages_only_first_two),
        "normalize": bool(normalize),
        "train": None,
        "ref": None,
    }

    progress_path = out_dir / "progress.json"
    progress = _load_progress(progress_path)

    # Train embeddings
    if not args.no_train:
        if not args.train_jsonl:
            raise ValueError("Provide --train_jsonl (or pass --no_train)")

        train_jsonl = Path(args.train_jsonl).expanduser().resolve()
        if not train_jsonl.exists():
            raise FileNotFoundError(str(train_jsonl))

        n_total = _count_lines(train_jsonl)
        if int(args.train_max_samples) > 0:
            n_total = min(n_total, int(args.train_max_samples))

        out_path = out_dir / "train_emb.npy"
        if out_path.exists() and progress.get("train_complete") is True:
            print(f"Train embeddings already complete: {out_path}")
        else:
            start = int(progress.get("train_next_idx") or 0)
            start = max(0, min(start, n_total))

            mode = "r+" if out_path.exists() else "w+"
            if mode == "w+":
                print(f"Allocating memmap {out_path} shape=({n_total}, {hidden_size}) dtype={out_dtype}")
            train_mm = open_memmap(out_path, mode=mode, dtype=out_dtype, shape=(n_total, hidden_size))

            eos = tokenizer.eos_token or ""
            iterator = _iter_jsonl_messages(train_jsonl, start=start, max_samples=int(args.train_max_samples))

            def texts_iter() -> Iterator[str]:
                for msgs in iterator:
                    if args.messages_only_first_two:
                        msgs = _messages_first_two(msgs)
                    txt = _format_messages(msgs, mode=str(args.format), eos_token=eos)
                    yield (str(args.doc_prefix) + txt) if args.doc_prefix else txt

            write_idx = start
            for batch in tqdm(_batched(texts_iter(), int(args.batch_size)), desc="Train emb"):
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                reps = _compute_emb(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pooling=str(args.pooling),
                    normalize=normalize,
                )
                reps_np = reps.detach().to(torch.float32).cpu().numpy()
                reps_np = reps_np.astype(out_dtype, copy=False)

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
            from transformers import AutoTokenizer as _AutoTokenizer

            base_tok = _AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
            if base_tok.pad_token is None:
                base_tok.pad_token = base_tok.eos_token
                base_tok.pad_token_id = base_tok.eos_token_id

            sys.path.insert(0, str(datelm_root))
            from minimal_multitask.data import DATASETS

            ref_name = str(args.ref_dataset_name)
            val_ds = DATASETS[ref_name](base_tok).get_all_test_prompts(
                num_samples=int(args.ref_num_samples),
                seed=int(args.ref_seed),
                prompt_only=False,
                response_only=False,
            )

            texts: list[str] = []
            for item in val_ds:
                input_ids = item.get("input_ids")
                if input_ids is None:
                    raise ValueError("Expected each ref item to contain input_ids")
                txt = base_tok.decode(input_ids, skip_special_tokens=True)
                texts.append((str(args.query_prefix) + txt) if args.query_prefix else txt)

            reps_all: list[np.ndarray] = []
            for batch in tqdm(_batched(texts, int(args.batch_size)), desc=f"Ref emb ({ref_name})"):
                enc = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                reps = _compute_emb(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pooling=str(args.pooling),
                    normalize=normalize,
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
