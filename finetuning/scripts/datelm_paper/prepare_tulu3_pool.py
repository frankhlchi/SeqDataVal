#!/usr/bin/env python
"""Prepare a deterministic 200k training pool (and optional val split) from Tulu3 JSONL.

This mirrors DATE-LM's `InstructJsonDataModule` sampling behavior:
  1) load_dataset("json", data_files=...)["train"]
  2) shuffle(seed)
  3) take `adjusted_size = int(pool_size / (1 - val_split))`
  4) train_test_split(test_size=val_split, seed=seed)

The resulting train split has size ~= `pool_size` and can be used as the
selection pool for paper-style experiments.

Example:
  python prepare_tulu3_pool.py \
    --input_jsonl /path/to/tulu_3_v3.9_unfiltered.jsonl \
    --out_train_jsonl /path/to/tulu3_200k_seed42_train.jsonl \
    --out_val_jsonl /path/to/tulu3_22k_seed42_val.jsonl \
    --seed 42 --pool_size 200000 --val_split 0.1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--out_train_jsonl", type=str, required=True)
    p.add_argument("--out_val_jsonl", type=str, default=None)
    p.add_argument("--out_meta_json", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool_size", type=int, default=200000)
    p.add_argument("--val_split", type=float, default=0.1)
    args = p.parse_args()

    from datasets import load_dataset

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(str(input_jsonl))

    out_train = Path(args.out_train_jsonl)
    out_train.parent.mkdir(parents=True, exist_ok=True)

    out_val = Path(args.out_val_jsonl) if args.out_val_jsonl else None
    if out_val:
        out_val.parent.mkdir(parents=True, exist_ok=True)

    out_meta = Path(args.out_meta_json) if args.out_meta_json else (out_train.parent / (out_train.stem + "_meta.json"))
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed)
    pool_size = int(args.pool_size)
    val_split = float(args.val_split)
    if not (0.0 < val_split < 1.0):
        raise ValueError("--val_split must be in (0, 1)")

    adjusted_size = int(pool_size / (1.0 - val_split))

    print("Loading dataset...")
    ds = load_dataset("json", data_files=str(input_jsonl), split="train")
    print(f"Loaded {len(ds)} rows")

    print(f"Shuffling with seed={seed}...")
    ds = ds.shuffle(seed=seed)

    print(f"Selecting adjusted_size={adjusted_size} (pool_size={pool_size}, val_split={val_split})...")
    ds = ds.select(range(min(adjusted_size, len(ds))))

    print("Splitting train/val...")
    split = ds.train_test_split(test_size=val_split, seed=seed)
    train_ds = split["train"]
    val_ds = split["test"]

    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")

    print(f"Writing train JSONL -> {out_train}")
    train_ds.to_json(str(out_train), orient="records", lines=True)

    if out_val:
        print(f"Writing val JSONL   -> {out_val}")
        val_ds.to_json(str(out_val), orient="records", lines=True)

    meta = {
        "input_jsonl": str(input_jsonl),
        "seed": seed,
        "pool_size": pool_size,
        "val_split": val_split,
        "adjusted_size": adjusted_size,
        "out_train_jsonl": str(out_train),
        "out_val_jsonl": str(out_val) if out_val else None,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Wrote meta -> {out_meta}")


if __name__ == "__main__":
    main()
