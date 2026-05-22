#!/usr/bin/env python
"""Compute SentenceTransformer embeddings for DATE-LM finetune pool + ref set.

This script supports BipCov "RAG embedding" ablations while keeping the DATE-LM
train/eval pipeline unchanged (train/finetune.py + minimal_multitask eval).

Outputs (under --out_dir):
  - train_emb.npy (optional)
  - ref_emb.npy (optional)
  - meta.json
  - progress.json (for resumable train embeddings)

Notes
- Train JSONL is expected to contain a `messages` field (OpenAI chat format).
- By default we mirror DATE-LM finetune preprocessing (`only_first_two=True`) for train.
- Ref texts are generated from DATE-LM's `minimal_multitask.data.DATASETS` (prompt+label),
  decoded using the base Llama tokenizer.
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


def _format_messages(messages: list[dict], mode: str) -> str:
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
                out += "<|assistant|>\n" + content + "\n"
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

    p.add_argument("--st_model", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dtype", type=str, default="float16")
    p.add_argument("--format", type=str, default="tulu_chat", choices=["tulu_chat", "plain"])
    p.add_argument("--doc_prefix", type=str, default="")
    p.add_argument("--query_prefix", type=str, default="")

    p.add_argument("--base_tokenizer", type=str, default="meta-llama/Llama-3.1-8B")

    # Train
    p.add_argument("--train_jsonl", type=str, default=None)
    p.add_argument("--train_max_samples", type=int, default=-1)
    p.add_argument("--messages_only_first_two", action="store_true")
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datelm_root = Path(args.datelm_root).expanduser().resolve()
    if not datelm_root.exists():
        raise FileNotFoundError(str(datelm_root))
    os.chdir(str(datelm_root))

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `sentence-transformers`. Install with: pip install sentence-transformers"
        ) from e

    from transformers import AutoTokenizer

    print(f"Loading SentenceTransformer: {args.st_model}")
    st = SentenceTransformer(args.st_model, device=args.device)
    dim = int(st.get_sentence_embedding_dimension())
    out_dtype = _to_numpy_dtype(args.out_dtype)

    print(f"Loading base tokenizer (for ref text decoding): {args.base_tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta: dict = {
        "st_model": args.st_model,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "out_dtype": str(out_dtype),
        "format": args.format,
        "doc_prefix": args.doc_prefix,
        "query_prefix": args.query_prefix,
        "base_tokenizer": args.base_tokenizer,
    }

    # Train embeddings
    if not args.no_train:
        if not args.train_jsonl:
            raise ValueError("Provide --train_jsonl (or pass --no_train)")
        train_jsonl = Path(args.train_jsonl).expanduser().resolve()
        if not train_jsonl.exists():
            raise FileNotFoundError(str(train_jsonl))

        out_path = out_dir / "train_emb.npy"
        progress_path = out_dir / "progress.json"
        progress = _load_progress(progress_path)
        if progress.get("train_complete") is True and out_path.exists():
            print(f"Train embeddings exist: {out_path}")
        else:
            n_total = _count_lines(train_jsonl)
            max_samples = int(args.train_max_samples)
            if max_samples > 0:
                n_total = min(n_total, max_samples)

            start = int(progress.get("train_next_idx") or 0)
            start = min(start, n_total)
            print(f"Computing train embeddings: total={n_total} start={start} dim={dim}")

            mode = "r+" if out_path.exists() else "w+"
            if mode == "w+":
                print(f"Allocating memmap {out_path} shape=({n_total}, {dim}) dtype={out_dtype}")
            train_mm = open_memmap(out_path, mode=mode, dtype=out_dtype, shape=(n_total, dim))

            def texts_iter() -> Iterator[str]:
                for msgs in _iter_jsonl_messages(train_jsonl, start=start, max_samples=max_samples):
                    if args.messages_only_first_two:
                        msgs = _messages_first_two(msgs)
                    txt = _format_messages(msgs, mode=args.format)
                    yield (args.doc_prefix + txt) if args.doc_prefix else txt

            write_idx = start
            for batch in tqdm(_batched(texts_iter(), int(args.batch_size)), desc="Train emb"):
                reps = st.encode(
                    batch,
                    batch_size=int(args.batch_size),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                reps = np.asarray(reps)
                if reps.ndim != 2 or reps.shape[1] != dim:
                    raise ValueError(f"Unexpected embedding shape: {reps.shape} (expected (*,{dim}))")
                end = write_idx + reps.shape[0]
                train_mm[write_idx:end] = reps.astype(out_dtype, copy=False)
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
            "out": str(out_path),
            "messages_only_first_two": bool(args.messages_only_first_two),
            "train_max_samples": int(args.train_max_samples),
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

            ref_name = str(args.ref_dataset_name)
            val_ds = DATASETS[ref_name](tokenizer).get_all_test_prompts(
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
                txt = tokenizer.decode(input_ids, skip_special_tokens=True)
                texts.append((args.query_prefix + txt) if args.query_prefix else txt)

            reps = st.encode(
                texts,
                batch_size=int(args.batch_size),
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            ref = np.asarray(reps, dtype=np.float32)
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
