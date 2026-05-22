#!/usr/bin/env python
"""Select top-k examples from a JSONL pool using a metric vector (or random scores).

This is a JSONL-friendly equivalent of DATE-LM's `methods/select_data.select` logic:
  - If --metrics_npy is not provided, we generate `rng.random(N)` with a fixed seed.
  - We select indices via `np.argpartition(scores, -k)[-k:]` (order is kept as returned).

Outputs:
  - selected JSONL
  - optional `selected_indices.npy`

Example (random baseline):
  python select_jsonl_topk.py \
    --pool_jsonl data/training_data/tulu3_200k_seed42_train.jsonl \
    --out_jsonl selected_data/paper_seed42/mmlu/random_10k.jsonl \
    --k 10000 --seed 42 \
    --out_indices_npy selected_data/paper_seed42/mmlu/random_indices.npy

Example (bipcov):
  python select_jsonl_topk.py \
    --pool_jsonl data/training_data/tulu3_200k_seed42_train.jsonl \
    --metrics_npy scores/paper_seed42/mmlu/bipcov_metrics.npy \
    --out_jsonl selected_data/paper_seed42/mmlu/bipcov_10k.jsonl \
    --k 10000 \
    --out_indices_npy selected_data/paper_seed42/mmlu/bipcov_indices.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool_jsonl", type=str, required=True)
    p.add_argument("--metrics_npy", type=str, default=None)
    p.add_argument("--k", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--out_indices_npy", type=str, default=None)
    args = p.parse_args()

    from datasets import load_dataset

    pool_jsonl = Path(args.pool_jsonl)
    if not pool_jsonl.exists():
        raise FileNotFoundError(str(pool_jsonl))

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_indices_npy = Path(args.out_indices_npy) if args.out_indices_npy else None
    if out_indices_npy:
        out_indices_npy.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=str(pool_jsonl), split="train")
    n = len(ds)
    k = int(min(args.k, n))

    if args.metrics_npy:
        metrics_path = Path(args.metrics_npy)
        if not metrics_path.exists():
            raise FileNotFoundError(str(metrics_path))
        scores = np.load(metrics_path)
        if scores.ndim != 1:
            scores = scores.reshape(-1)
        if len(scores) != n:
            raise ValueError(f"metrics length mismatch: len(scores)={len(scores)} vs N={n}")
    else:
        rng = np.random.default_rng(int(args.seed))
        scores = rng.random(n, dtype=np.float32)

    selected_indices = np.argpartition(scores, -k)[-k:]
    selected_indices = selected_indices.astype(np.int64)

    if out_indices_npy:
        np.save(out_indices_npy, selected_indices)

    subset = ds.select(selected_indices.tolist())
    subset.to_json(str(out_jsonl), orient="records", lines=True)

    print(f"Pool: {pool_jsonl} (N={n})")
    print(f"Selected: k={k} -> {out_jsonl}")
    if out_indices_npy:
        print(f"Indices: {out_indices_npy}")


if __name__ == "__main__":
    main()
