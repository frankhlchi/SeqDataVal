#!/usr/bin/env python
"""Select top-k examples from a JSONL pool using Gumbel-Top-k (diversity sampling).

This implements DATE-LM's RDS+ (Rep-Sim with Diversity Sampling) selection logic:
  1. Load metric scores (e.g., RepSim scores)
  2. Z-score normalize the scores
  3. Scale by Gumbel temperature
  4. Add Gumbel noise for diversity
  5. Select top-k

The Gumbel-Top-k algorithm adds stochastic diversity to greedy top-k selection,
ensuring selected samples are both high-quality and diverse.

Outputs:
  - selected JSONL
  - optional `selected_indices.npy`

Example (RDS+ with gumbel_temp=0.5):
  python select_jsonl_gumbel_topk.py \
    --pool_jsonl data/training_data/tulu3_200k_seed42_train.jsonl \
    --metrics_npy scores/paper_seed42/mmlu/repsim_metrics.npy \
    --out_jsonl selected_data/paper_seed42/mmlu/rds_plus_10k.jsonl \
    --k 10000 \
    --gumbel_temp 0.5 \
    --out_indices_npy selected_data/paper_seed42/mmlu/rds_plus_indices.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def gumbel_topk_select(
    scores: np.ndarray,
    k: int,
    gumbel_temp: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Select top-k indices using Gumbel-Top-k algorithm.

    This is the diversity sampling method used in DATE-LM's RDS+.

    Args:
        scores: 1D array of metric scores (higher = better)
        k: number of samples to select
        gumbel_temp: Gumbel temperature (lower = more deterministic, higher = more random)
        seed: random seed for Gumbel noise

    Returns:
        Selected indices (1D array of size k)
    """
    # 1. Z-score normalize
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    if std_score < 1e-8:
        # Avoid division by zero
        normalized = scores - mean_score
    else:
        normalized = (scores - mean_score) / std_score

    # 2. Scale by Gumbel temperature
    scaled = normalized / gumbel_temp

    # 3. Add Gumbel noise
    rng = np.random.default_rng(seed)
    gumbel_noise = rng.gumbel(size=len(scores))
    perturbed = scaled + gumbel_noise

    # 4. Select top-k
    selected_indices = np.argpartition(perturbed, -k)[-k:]

    return selected_indices.astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool_jsonl", type=str, required=True,
                   help="Path to the training pool JSONL file")
    p.add_argument("--metrics_npy", type=str, required=True,
                   help="Path to the metric scores (e.g., RepSim scores)")
    p.add_argument("--k", type=int, default=10000,
                   help="Number of samples to select")
    p.add_argument("--gumbel_temp", type=float, default=0.5,
                   help="Gumbel temperature (0.5 is DATE-LM default for RDS+)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for Gumbel noise")
    p.add_argument("--out_jsonl", type=str, required=True,
                   help="Output path for selected JSONL")
    p.add_argument("--out_indices_npy", type=str, default=None,
                   help="Optional output path for selected indices")
    args = p.parse_args()

    from datasets import load_dataset

    pool_jsonl = Path(args.pool_jsonl)
    if not pool_jsonl.exists():
        raise FileNotFoundError(str(pool_jsonl))

    metrics_path = Path(args.metrics_npy)
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    out_indices_npy = Path(args.out_indices_npy) if args.out_indices_npy else None
    if out_indices_npy:
        out_indices_npy.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset("json", data_files=str(pool_jsonl), split="train")
    n = len(ds)
    k = int(min(args.k, n))

    # Load scores
    scores = np.load(metrics_path)
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if len(scores) != n:
        raise ValueError(f"metrics length mismatch: len(scores)={len(scores)} vs N={n}")

    print(f"Gumbel-Top-k selection:")
    print(f"  - Pool size: {n}")
    print(f"  - Selection size: {k}")
    print(f"  - Gumbel temperature: {args.gumbel_temp}")
    print(f"  - Seed: {args.seed}")
    print(f"  - Score stats: mean={scores.mean():.4f}, std={scores.std():.4f}, "
          f"min={scores.min():.4f}, max={scores.max():.4f}")

    # Gumbel-Top-k selection
    selected_indices = gumbel_topk_select(
        scores=scores,
        k=k,
        gumbel_temp=args.gumbel_temp,
        seed=args.seed,
    )

    # Save indices
    if out_indices_npy:
        np.save(out_indices_npy, selected_indices)

    # Save selected JSONL
    subset = ds.select(selected_indices.tolist())
    subset.to_json(str(out_jsonl), orient="records", lines=True)

    # Print statistics
    selected_scores = scores[selected_indices]
    print(f"\nSelected subset statistics:")
    print(f"  - Mean score: {selected_scores.mean():.4f} (vs pool mean {scores.mean():.4f})")
    print(f"  - Std score: {selected_scores.std():.4f}")
    print(f"  - Min score: {selected_scores.min():.4f}")
    print(f"  - Max score: {selected_scores.max():.4f}")
    print(f"\nOutput:")
    print(f"  - Selected JSONL: {out_jsonl}")
    if out_indices_npy:
        print(f"  - Indices: {out_indices_npy}")


if __name__ == "__main__":
    main()
