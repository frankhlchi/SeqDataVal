#!/usr/bin/env python3
"""Generate a fixed random metric file for DATE-LM selection.

Why:
  - DATE-LM `train/finetune.py` uses `select_from_size=200000`, which can be
    off-by-one vs the actual HF split size depending on datasets versions.
  - Providing an explicit metrics.npy aligned to the actual train split length
    avoids out-of-range indices while still implementing random top-k selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _infer_n(reference_metrics: Path) -> int:
    arr = np.load(reference_metrics, mmap_mode="r")
    return int(arr.reshape(-1).shape[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_metrics", type=Path, required=True, help="Output path for random metrics .npy")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help="Length of metrics array (recommended: actual train split size).",
    )
    p.add_argument(
        "--reference_metrics",
        type=Path,
        default=None,
        help="Infer length from an existing metrics.npy (e.g., repsim_metrics.npy).",
    )
    args = p.parse_args()

    if args.n is None:
        if args.reference_metrics is None:
            raise SystemExit("Provide either --n or --reference_metrics.")
        n = _infer_n(args.reference_metrics)
    else:
        n = int(args.n)

    rng = np.random.default_rng(int(args.seed))
    metrics = rng.random(n, dtype=np.float32)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_metrics, metrics)
    print(f"[OK] wrote: {args.out_metrics} shape={metrics.shape} seed={args.seed}")


if __name__ == "__main__":
    main()

