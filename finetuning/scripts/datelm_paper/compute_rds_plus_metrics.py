#!/usr/bin/env python3
"""Compute an RDS+ *proxy* metric file by Gumbel-perturbing Rep-Sim scores.

Important: The DATE-LM paper (arXiv:2507.09424) describes **RDS+** as
"position-weighted mean pool of the last hidden layer states of all tokens"
(see `finetuning/arXiv-2507.09424v2.tar.gz` → `sections/appendix_data_attribution_methods.tex`).
This script does **not** implement that embedding-based RDS+ baseline; it only perturbs an
existing Rep-Sim metric vector.

We keep this script to reproduce our historical `rds_plus` runs (Rep-Sim + Gumbel),
and to make it explicit what was actually executed.

For the **paper-faithful RDS+** (position-weighted mean pooling over tokens), use:
  - `compute_llm_last_token_embeddings.py --pooling weighted_mean` (train + ref)
  - `compute_repsim_metrics_from_emb.py` to write `scores/.../rds_plus_metrics.npy`

Implementation here:
  1) load Rep-Sim scores (1D; higher is better)
  2) Z-score normalize
  3) scale by 1 / gumbel_temp
  4) add seeded Gumbel noise
  5) save the perturbed scores as float32 .npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repsim_metrics", type=Path, required=True, help="Path to Rep-Sim metrics .npy")
    p.add_argument("--out_metrics", type=Path, required=True, help="Output path for RDS+ metrics .npy")
    p.add_argument("--gumbel_temp", type=float, default=0.5, help="Gumbel temperature (DATE-LM default: 0.5)")
    p.add_argument("--seed", type=int, default=42, help="Seed for Gumbel noise (DATE-LM uses 42)")
    args = p.parse_args()

    repsim = np.load(args.repsim_metrics)
    if repsim.ndim != 1:
        repsim = repsim.reshape(-1)

    mean = float(np.mean(repsim))
    std = float(np.std(repsim))
    if std < 1e-8:
        normalized = repsim - mean
    else:
        normalized = (repsim - mean) / std

    scaled = normalized / float(args.gumbel_temp)

    rng = np.random.default_rng(int(args.seed))
    gumbel_noise = rng.gumbel(size=scaled.shape[0])
    out = (scaled + gumbel_noise).astype(np.float32)

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_metrics, out)

    print(f"[OK] wrote: {args.out_metrics}")
    print(f"  shape: {out.shape}")
    print(f"  repsim mean/std: {mean:.6f}/{std:.6f}")
    print(f"  out mean/std:    {float(out.mean()):.6f}/{float(out.std()):.6f}")


if __name__ == "__main__":
    main()
