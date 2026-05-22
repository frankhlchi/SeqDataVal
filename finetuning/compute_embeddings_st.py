#!/usr/bin/env python
"""Compute dense embeddings for train pool and ref set using SentenceTransformers.

This helper is intentionally simple:
- reads local JSONL files
- extracts a text field (instruction / prompt / question)
- computes embeddings with a SentenceTransformer model (e.g., BGE / E5)

Outputs:
- train_emb.npy  (N,d)
- ref_emb.npy    (M,d)

NOTE: this script downloads models from Hugging Face unless they are already cached.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def read_jsonl(path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_items is not None and len(items) >= max_items:
                break
    return items


def extract_text(obj: Dict[str, Any], text_key: Optional[str] = None) -> str:
    """Best-effort extraction for common instruction-tuning JSONL formats."""
    if text_key is not None:
        if text_key not in obj:
            raise KeyError(f"text_key='{text_key}' not found in object keys={list(obj.keys())[:20]}")
        return str(obj[text_key])

    # common keys
    for k in ["instruction", "prompt", "question", "text"]:
        if k in obj and obj[k] is not None:
            return str(obj[k])

    # OpenAI chat-like messages
    if "messages" in obj and isinstance(obj["messages"], list):
        # prefer the first user message
        for m in obj["messages"]:
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                return str(m["content"])
        # fallback: concatenate
        parts = []
        for m in obj["messages"]:
            if isinstance(m, dict) and m.get("content"):
                parts.append(str(m["content"]))
        if parts:
            return "\n".join(parts)

    # last resort
    return json.dumps(obj, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--ref-jsonl", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--train-text-key", type=str, default=None)
    parser.add_argument("--ref-text-key", type=str, default=None)

    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-ref", type=int, default=None)

    parser.add_argument("--model-name", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument(
        "--query-prefix",
        type=str,
        default="Represent this question for retrieving relevant training data: ",
        help="Prefix added to ref texts (query).",
    )
    parser.add_argument("--no-prefix", action="store_true", help="Do not add prefix to ref texts")

    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before saving")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = read_jsonl(Path(args.train_jsonl), max_items=args.max_train)
    ref_items = read_jsonl(Path(args.ref_jsonl), max_items=args.max_ref)

    train_texts = [extract_text(x, args.train_text_key) for x in train_items]
    ref_texts_raw = [extract_text(x, args.ref_text_key) for x in ref_items]

    if args.no_prefix:
        ref_texts = ref_texts_raw
    else:
        ref_texts = [args.query_prefix + t for t in ref_texts_raw]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name, device=args.device)

    print(f"Encoding train: {len(train_texts)} examples")
    train_emb = model.encode(train_texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)

    print(f"Encoding ref: {len(ref_texts)} examples")
    ref_emb = model.encode(ref_texts, batch_size=args.batch_size, convert_to_numpy=True, show_progress_bar=True)

    train_emb = np.asarray(train_emb, dtype=np.float32)
    ref_emb = np.asarray(ref_emb, dtype=np.float32)

    if args.normalize:
        train_emb /= np.linalg.norm(train_emb, axis=1, keepdims=True).clip(min=1e-12)
        ref_emb /= np.linalg.norm(ref_emb, axis=1, keepdims=True).clip(min=1e-12)

    np.save(out_dir / "train_emb.npy", train_emb)
    np.save(out_dir / "ref_emb.npy", ref_emb)

    print("Saved:")
    print(" ", out_dir / "train_emb.npy", train_emb.shape)
    print(" ", out_dir / "ref_emb.npy", ref_emb.shape)


if __name__ == "__main__":
    main()
