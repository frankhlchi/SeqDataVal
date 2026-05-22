#!/usr/bin/env python3
"""Compute Rep-Sim metrics from precomputed embeddings.

Rep-Sim score for each train example is the mean similarity to the reference set:
  score[i] = mean_j <train_emb[i], ref_emb[j]>

Notes:
  - This is chunked over train rows and supports `train_emb.npy` as an mmap.
  - For DATE-LM Table 3 reproduction, make sure the embeddings were produced with
    the same representation extraction as the scorer you want to match.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_dtype(name: str):
    name = name.lower().strip()
    if name in {"fp16", "float16"}:
        return "float16"
    if name in {"bf16", "bfloat16"}:
        return "bfloat16"
    if name in {"fp32", "float32"}:
        return "float32"
    raise ValueError(f"Unsupported dtype: {name} (expected fp16|bf16|fp32)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_emb", type=Path, required=True, help="Path to train_emb.npy (N,d)")
    p.add_argument("--ref_emb", type=Path, required=True, help="Path to ref_emb.npy (M,d)")
    p.add_argument("--out", type=Path, required=True, help="Output metrics.npy (N,)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        help="Computation dtype for matmul on the chosen device (fp16|bf16|fp32).",
    )
    p.add_argument("--batch_rows", type=int, default=50000, help="Rows per chunk for train_emb")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    train = np.load(args.train_emb, mmap_mode="r")
    ref = np.load(args.ref_emb)
    if train.ndim != 2 or ref.ndim != 2:
        raise ValueError("train_emb and ref_emb must both be 2D arrays")
    if train.shape[1] != ref.shape[1]:
        raise ValueError(f"Dim mismatch: train {train.shape} vs ref {ref.shape}")

    if args.out.exists() and not args.overwrite:
        print(f"[SKIP] exists: {args.out}")
        return

    dtype = _parse_dtype(args.dtype)
    batch_rows = max(1, int(args.batch_rows))

    out = np.empty((train.shape[0],), dtype=np.float32)

    use_torch = True
    try:
        import torch
    except Exception:
        use_torch = False

    if use_torch:
        device = torch.device(args.device)
        ref_t = torch.from_numpy(ref).to(device=device)
        if dtype == "float16":
            ref_t = ref_t.to(dtype=torch.float16)
        elif dtype == "bfloat16":
            ref_t = ref_t.to(dtype=torch.bfloat16)
        else:
            ref_t = ref_t.to(dtype=torch.float32)

        for start in range(0, train.shape[0], batch_rows):
            end = min(train.shape[0], start + batch_rows)
            chunk = np.array(train[start:end])  # materialize memmap slice
            chunk_t = torch.from_numpy(chunk).to(device=device, dtype=ref_t.dtype)
            with torch.inference_mode():
                scores = (chunk_t @ ref_t.T).mean(dim=1).to(dtype=torch.float32).cpu().numpy()
            out[start:end] = scores
    else:
        # CPU fallback (slow but works everywhere)
        ref_f = ref.astype(np.float32, copy=False)
        for start in range(0, train.shape[0], batch_rows):
            end = min(train.shape[0], start + batch_rows)
            chunk = np.array(train[start:end], dtype=np.float32)
            out[start:end] = (chunk @ ref_f.T).mean(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)

    print(f"[OK] wrote: {args.out} shape={out.shape}")


if __name__ == "__main__":
    main()
