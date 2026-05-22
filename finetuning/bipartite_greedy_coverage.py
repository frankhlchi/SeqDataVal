#!/usr/bin/env python
"""Bipartite Greedy Coverage baseline (DATE-LM style).

Given:
  - train embeddings: (N, d)
  - reference embeddings: (M, d)

We:
  1) build a bipartite graph via sim(x_i, r_j) >= tau
  2) run **lazy greedy max-coverage** to pick k examples
  3) output `metrics.npy` (shape (N,)) suitable for DATE-LM top-k selection.

This script is designed to:
  - run a tiny CPU demo
  - scale to DATE-LM-ish settings (e.g., N=200K, M<=100) when used with GPU.

Notes:
  - Greedy coverage is monotone submodular; lazy greedy yields the same greedy
    solution but faster.
  - Embeddings should be L2-normalized; you can pass --normalize to enforce it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


# local popcount LUT (numpy 1.24 has no vectorized bit_count)
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_bytes_1d(x: np.ndarray) -> int:
    return int(_POPCOUNT_LUT[x.astype(np.uint8, copy=False)].sum())


def _pack_bool_matrix(adj: np.ndarray) -> np.ndarray:
    if adj.dtype != np.bool_:
        adj = adj.astype(np.bool_, copy=False)
    return np.packbits(adj, axis=1, bitorder="little")


def lazy_greedy_max_coverage_packed(packed_neighbors: np.ndarray, n_ref: int, k: int) -> list[int]:
    """Lazy greedy max-coverage over packed uint8 bitsets.

    Returns exactly k indices (deterministic fill for the rest).
    """
    import heapq

    if packed_neighbors.ndim != 2:
        raise ValueError("packed_neighbors must be 2D")
    if packed_neighbors.dtype != np.uint8:
        packed_neighbors = packed_neighbors.astype(np.uint8, copy=False)

    n_train, n_bytes = packed_neighbors.shape
    if k > n_train:
        raise ValueError(f"k={k} > n_train={n_train}")

    # uncovered bitmask (packed)
    uncovered = np.full(n_bytes, 255, dtype=np.uint8)
    if n_ref % 8 != 0:
        uncovered[-1] = (1 << (n_ref % 8)) - 1

    gains0 = _POPCOUNT_LUT[packed_neighbors].sum(axis=1).astype(np.int64)
    heap: list[Tuple[int, int]] = [(-int(g), int(i)) for i, g in enumerate(gains0)]
    heapq.heapify(heap)

    chosen = np.zeros(n_train, dtype=np.bool_)
    selected: list[int] = []

    while len(selected) < k and heap:
        if -heap[0][0] <= 0:
            break

        neg_est, idx = heapq.heappop(heap)
        if chosen[idx]:
            continue

        gain_true = _popcount_bytes_1d(packed_neighbors[idx] & uncovered)
        best_est_next = -heap[0][0] if heap else 0

        if gain_true >= best_est_next:
            selected.append(idx)
            chosen[idx] = True
            uncovered &= ~packed_neighbors[idx]
            if not uncovered.any():
                break
        else:
            heapq.heappush(heap, (-int(gain_true), idx))

    if len(selected) < k:
        remaining_idx = np.flatnonzero(~chosen)
        fill = remaining_idx[: (k - len(selected))].astype(int).tolist()
        selected.extend(fill)

    return selected


def coverage_from_packed(packed_neighbors: np.ndarray, selected: Iterable[int], n_ref: int) -> float:
    n_bytes = packed_neighbors.shape[1]
    covered = np.zeros(n_bytes, dtype=np.uint8)
    for idx in selected:
        covered |= packed_neighbors[int(idx)]
    covered_bits = _popcount_bytes_1d(covered)
    return float(covered_bits) / float(n_ref)


def learn_threshold_pilot(
    train_emb: np.ndarray,
    ref_emb: np.ndarray,
    device: str,
    batch_size: int,
    k: int,
    pilot_size: int,
    pilot_k: int,
    tau_grid: np.ndarray,
    seed: int,
) -> Tuple[float, dict]:
    """Learn similarity threshold using a pilot subset.

    We evaluate tau candidates by greedy coverage on a subset of training points.
    """
    rng = np.random.default_rng(seed)
    n_train = train_emb.shape[0]
    pilot_size = min(pilot_size, n_train)

    # sample pilot indices
    idx = rng.choice(n_train, size=pilot_size, replace=False)
    sample = np.asarray(train_emb[idx], dtype=np.float32)

    # compute similarities (pilot_size, n_ref)
    sim = compute_similarities(sample, ref_emb, device=device, batch_size=batch_size)

    n_ref = ref_emb.shape[0]

    best_tau = float(tau_grid[0])
    best_obj = -1e9
    best_diag = {}

    # choose pilot_k relative to target k
    pilot_k = int(min(pilot_k, pilot_size, max(10, k // 50)))

    for tau in tau_grid:
        adj = sim >= float(tau)
        packed = _pack_bool_matrix(adj)

        sel = lazy_greedy_max_coverage_packed(packed, n_ref=n_ref, k=pilot_k)
        cov = coverage_from_packed(packed, sel, n_ref=n_ref)
        density = float(adj.mean())

        # heuristic objective: high coverage, discourage ultra-dense graphs
        # (dense graphs reduce discrimination, and coverage becomes trivial)
        # target density roughly in [0.01, 0.10] for M<=100 is often reasonable.
        target = 0.05
        penalty = abs(np.log((density + 1e-9) / target))
        obj = cov - 0.15 * penalty

        if obj > best_obj:
            best_obj = obj
            best_tau = float(tau)
            best_diag = {
                "pilot_size": int(pilot_size),
                "pilot_k": int(pilot_k),
                "coverage": float(cov),
                "graph_density": float(density),
                "density_penalty": float(penalty),
                "objective": float(obj),
            }

    return best_tau, best_diag


def compute_similarities(
    train_emb: np.ndarray,
    ref_emb: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Compute cosine similarity matrix in batches (optionally on GPU via torch)."""
    try:
        import torch
        import torch.nn.functional as F

        use_torch = True
    except Exception:
        use_torch = False

    if not use_torch or device == "cpu":
        # numpy fallback (OK for small runs)
        return train_emb @ ref_emb.T

    # torch path
    dev = torch.device(device)
    ref_t = torch.from_numpy(ref_emb).to(dev)
    ref_t = F.normalize(ref_t, dim=1)

    out = np.zeros((train_emb.shape[0], ref_emb.shape[0]), dtype=np.float32)

    for start in range(0, train_emb.shape[0], batch_size):
        end = min(start + batch_size, train_emb.shape[0])
        batch = torch.from_numpy(np.asarray(train_emb[start:end], dtype=np.float32)).to(dev)
        batch = F.normalize(batch, dim=1)
        sim = batch @ ref_t.T
        out[start:end] = sim.detach().cpu().numpy().astype(np.float32, copy=False)

    return out


def build_packed_neighbors_streaming(
    train_emb_path: Path,
    ref_emb: np.ndarray,
    tau: float,
    device: str,
    batch_size: int,
    normalize: bool,
) -> Tuple[np.ndarray, dict]:
    """Build packed neighbors for all training examples without storing full sim matrix."""
    try:
        import torch
        import torch.nn.functional as F

        use_torch = True
    except Exception:
        use_torch = False

    train_emb = np.load(train_emb_path, mmap_mode="r")
    n_train, d = train_emb.shape
    n_ref = ref_emb.shape[0]
    n_bytes = int((n_ref + 7) // 8)

    packed_all = np.zeros((n_train, n_bytes), dtype=np.uint8)

    edges = 0

    if use_torch and device != "cpu":
        dev = torch.device(device)
        ref_t = torch.from_numpy(ref_emb.astype(np.float32, copy=False)).to(dev)
        if normalize:
            ref_t = F.normalize(ref_t, dim=1)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_np = np.asarray(train_emb[start:end], dtype=np.float32)
            batch_t = torch.from_numpy(batch_np).to(dev)
            if normalize:
                batch_t = F.normalize(batch_t, dim=1)
            sim = batch_t @ ref_t.T
            adj = (sim >= tau).detach().cpu().numpy().astype(np.bool_)
            edges += int(adj.sum())
            packed_all[start:end] = np.packbits(adj, axis=1, bitorder="little")
    else:
        # CPU / numpy path (only intended for small runs)
        ref_np = ref_emb.astype(np.float32, copy=False)
        if normalize:
            ref_np = _l2_normalize_rows(ref_np)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch = np.asarray(train_emb[start:end], dtype=np.float32)
            if normalize:
                batch = _l2_normalize_rows(batch)
            sim = batch @ ref_np.T
            adj = (sim >= tau)
            edges += int(adj.sum())
            packed_all[start:end] = np.packbits(adj.astype(np.bool_), axis=1, bitorder="little")

    meta = {
        "n_train": int(n_train),
        "n_ref": int(n_ref),
        "dim": int(d),
        "threshold": float(tau),
        "graph_density": float(edges / max(1, (n_train * n_ref))),
        "n_edges": int(edges),
    }
    return packed_all, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-emb", type=str, required=True, help="Path to train_emb.npy (N,d)")
    parser.add_argument("--ref-emb", type=str, required=True, help="Path to ref_emb.npy (M,d)")
    parser.add_argument("--k", type=int, required=True, help="Number of training points to select")

    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold tau")
    parser.add_argument("--learn-threshold", action="store_true", help="Grid search tau on a pilot subset")

    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings before similarity")

    parser.add_argument("--pilot-size", type=int, default=20000)
    parser.add_argument("--pilot-k", type=int, default=1000)
    parser.add_argument("--tau-min", type=float, default=0.30)
    parser.add_argument("--tau-max", type=float, default=0.80)
    parser.add_argument("--tau-steps", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_emb_path = Path(args.train_emb)
    ref_emb_path = Path(args.ref_emb)

    ref_emb = np.load(ref_emb_path)
    ref_emb = np.asarray(ref_emb, dtype=np.float32)
    if args.normalize:
        ref_emb = _l2_normalize_rows(ref_emb)

    # learn threshold if needed
    tau = args.threshold
    diag = {}
    if tau is None:
        if not args.learn_threshold:
            raise ValueError("Provide --threshold or set --learn-threshold")

        tau_grid = np.linspace(args.tau_min, args.tau_max, args.tau_steps, dtype=np.float32)
        # For pilot we load train_emb as memmap and sample from it.
        train_emb_mmap = np.load(train_emb_path, mmap_mode="r")
        if args.normalize:
            # normalization will be applied in compute_similarities
            pass
        tau, diag = learn_threshold_pilot(
            train_emb=train_emb_mmap,
            ref_emb=ref_emb,
            device=args.device,
            batch_size=min(args.batch_size, 4096),
            k=args.k,
            pilot_size=args.pilot_size,
            pilot_k=args.pilot_k,
            tau_grid=tau_grid,
            seed=args.seed,
        )

    # build packed neighbor sets for all points
    packed, meta = build_packed_neighbors_streaming(
        train_emb_path=train_emb_path,
        ref_emb=ref_emb,
        tau=float(tau),
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
    )

    # select
    selected = lazy_greedy_max_coverage_packed(packed, n_ref=meta["n_ref"], k=args.k)
    cov = coverage_from_packed(packed, selected, n_ref=meta["n_ref"])

    # scores (DATE-LM top-k compatible)
    scores = np.zeros((meta["n_train"],), dtype=np.float32)
    # higher = earlier in greedy order
    for r, idx in enumerate(selected):
        scores[int(idx)] = float(len(selected) - r)

    # save
    np.save(out_dir / "metrics.npy", scores)
    (out_dir / "selected_indices.json").write_text(json.dumps(selected, indent=2))

    meta_out = {
        **meta,
        "k": int(args.k),
        "coverage": float(cov),
        "learn_threshold": bool(args.learn_threshold),
        "threshold_diagnostics": diag,
        "device": str(args.device),
        "batch_size": int(args.batch_size),
        "normalized": bool(args.normalize),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    print("Saved:")
    print("  ", out_dir / "metrics.npy")
    print("  ", out_dir / "selected_indices.json")
    print("  ", out_dir / "meta.json")
    print(f"coverage={cov:.4f}  density={meta['graph_density']:.6f}  tau={tau:.4f}")


if __name__ == "__main__":
    main()
