"""Bipartite Greedy Coverage (BipCov) baseline using precomputed embeddings.

Scheme-1 integration (recommended):
  - Compute and save embeddings externally:
      train_emb.npy: shape (N_train, d)
      ref_emb.npy:   shape (N_ref, d)
  - This script reads these embeddings and produces a DATE-LM-compatible
    metric file (a 1D score vector) such that selecting top-k by score yields
    the greedy coverage subset.

Why scores instead of directly outputting indices?
  DATE-LM fine-tuning pipeline expects a metric_path (.npy) of length N_train.
  It then selects top-k indices via `methods/select_data.select`.

Notes:
  - This implementation is CPU-friendly.
  - For large N, the expensive part is similarity computation. You can enable
    torch/cuda matmul with --device cuda if desired.

Example (toy):
  python methods/bipcov/probe_bipcov_from_emb.py \
    --train_emb embeddings/train_emb.npy \
    --ref_emb embeddings/mmlu_ref_emb.npy \
    --out scores/mmlu_shots_bipcov.npy \
    --k_max 10000 --threshold 0.45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _maybe_load(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return arr


def _compute_sim(
    train_emb: np.ndarray,
    ref_emb: np.ndarray,
    device: str = "cpu",
    batch_rows: int = 50000,
) -> np.ndarray:
    """Compute cosine similarity matrix between L2-normalized embeddings.

    Returns sim of shape (N_train, N_ref).
    """

    train_emb = train_emb.astype(np.float32, copy=False)
    ref_emb = ref_emb.astype(np.float32, copy=False)
    train_emb = _l2_normalize(train_emb)
    ref_emb = _l2_normalize(ref_emb)

    if device.startswith("cuda"):
        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                "--device cuda requires PyTorch. Install torch or use --device cpu."
            ) from e

        dev = torch.device(device)
        ref_t = torch.from_numpy(ref_emb).to(dev)
        sims: List[np.ndarray] = []
        for start in range(0, train_emb.shape[0], batch_rows):
            end = min(train_emb.shape[0], start + batch_rows)
            chunk = torch.from_numpy(train_emb[start:end]).to(dev)
            sim_chunk = (chunk @ ref_t.T).float().cpu().numpy()
            sims.append(sim_chunk)
        return np.concatenate(sims, axis=0)

    # CPU path
    return train_emb @ ref_emb.T


def _build_adj_lists_threshold(
    sim: np.ndarray,
    threshold: float,
    ensure_ref_coverage: bool = True,
) -> List[np.ndarray]:
    """Build adjacency lists per ref j: list of train indices i with sim[i,j] >= threshold."""
    n_train, n_ref = sim.shape
    adj: List[np.ndarray] = []
    for j in range(n_ref):
        idx = np.flatnonzero(sim[:, j] >= threshold)
        if ensure_ref_coverage and idx.size == 0:
            # force at least one neighbor
            idx = np.array([int(np.argmax(sim[:, j]))], dtype=np.int64)
        adj.append(idx.astype(np.int64, copy=False))
    return adj


def _build_adj_lists_topl(sim: np.ndarray, top_l: int) -> List[np.ndarray]:
    """Build adjacency lists per ref j: top-L most similar training indices."""
    if top_l <= 0:
        raise ValueError("top_l must be positive")
    n_train, n_ref = sim.shape
    top_l = int(min(top_l, n_train))
    adj: List[np.ndarray] = []
    for j in range(n_ref):
        col = sim[:, j]
        # argpartition gives arbitrary order; we sort for determinism
        idx = np.argpartition(col, -top_l)[-top_l:]
        idx = idx[np.argsort(-col[idx])]
        adj.append(idx.astype(np.int64, copy=False))
    return adj


def greedy_max_coverage_from_adj(
    adj_by_ref: List[np.ndarray],
    sim_row_getter,
    n_train: int,
    k_max: int,
    n_ref: int,
) -> Tuple[List[int], np.ndarray]:
    """Greedy max-coverage using degree updates via adjacency lists.

    Returns:
      - selected sequence (length <= k_max)
      - covered boolean array of shape (n_ref,)

    Implementation detail:
      We maintain deg[i] = number of *uncovered* refs covered by train i.
      When a ref becomes covered, we decrement deg for all train nodes
      connected to that ref (one vectorized op).
    """

    deg = np.zeros(n_train, dtype=np.int32)
    for idxs in adj_by_ref:
        deg[idxs] += 1

    selected_mask = np.zeros(n_train, dtype=np.bool_)
    covered = np.zeros(n_ref, dtype=np.bool_)
    selected: List[int] = []

    # At most n_ref iterations can improve binary coverage.
    while len(selected) < k_max:
        best = int(np.argmax(deg))
        best_gain = int(deg[best])
        if best_gain <= 0:
            break
        selected.append(best)
        selected_mask[best] = True
        deg[best] = -1  # exclude

        # Which refs does `best` newly cover?
        sim_best = sim_row_getter(best)  # shape (n_ref,)
        newly = np.flatnonzero(~covered & sim_best)
        if newly.size == 0:
            continue

        for j in newly.tolist():
            covered[j] = True
            # decrement candidates connected to ref j
            idxs = adj_by_ref[j]
            deg[idxs] -= 1
            # keep selected excluded
            deg[selected_mask] = -1

        if covered.all():
            break

    return selected, covered


def build_scores_from_sequence(
    base_scores: np.ndarray,
    sequence: List[int],
    k_max: int,
) -> np.ndarray:
    """Ensure greedy-selected items are top-ranked by boosting their scores above others."""
    scores = base_scores.astype(np.float32, copy=True)
    # Normalize base scores to [0, 1]
    smin, smax = float(scores.min()), float(scores.max())
    if smax > smin:
        scores = (scores - smin) / (smax - smin)
    else:
        scores[:] = 0.0

    # Boost greedy picks above 1.0, keep their relative order
    for pos, idx in enumerate(sequence):
        # earlier => larger
        scores[idx] = 1.0 + float(k_max - pos) / float(k_max)
    return scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_emb", type=str, required=True, help="Path to train_emb.npy")
    p.add_argument("--ref_emb", type=str, required=True, help="Path to ref_emb.npy")
    p.add_argument(
        "--out",
        type=str,
        required=True,
        help=(
            "Output path. If it ends with .npy, writes a single file. "
            "If it is a directory, writes <out>/metrics.npy."
        ),
    )
    p.add_argument("--k_max", type=int, default=10000, help="Max selection size to encode in scores")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold for edges. If not set, uses --target_density quantile.",
    )
    p.add_argument(
        "--target_density",
        type=float,
        default=0.02,
        help="When --threshold is None, choose tau by (1-density) quantile of similarities.",
    )
    p.add_argument(
        "--top_l",
        type=int,
        default=None,
        help="If set, connect each ref to its top-L training neighbors (ignores threshold).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda (optionally cuda:0). Only used for similarity computation.",
    )
    p.add_argument(
        "--batch_rows",
        type=int,
        default=50000,
        help="Row chunk size when computing similarity on GPU.",
    )
    p.add_argument(
        "--ensure_ref_coverage",
        action="store_true",
        help="Ensure each ref has at least one neighbor under threshold.",
    )
    p.add_argument(
        "--save_selected",
        type=str,
        default=None,
        help="Optional path to save selected indices JSON.",
    )
    args = p.parse_args()

    train_emb = _maybe_load(args.train_emb)
    ref_emb = _maybe_load(args.ref_emb)
    if train_emb.ndim != 2 or ref_emb.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays")
    if train_emb.shape[1] != ref_emb.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: train={train_emb.shape}, ref={ref_emb.shape}"
        )
    n_train, d = train_emb.shape
    n_ref = ref_emb.shape[0]
    k_max = int(min(args.k_max, n_train))

    print(f">> train_emb: {train_emb.shape}  ref_emb: {ref_emb.shape}  k_max={k_max}")

    # Compute similarities
    sim = _compute_sim(train_emb, ref_emb, device=args.device, batch_rows=args.batch_rows)
    sim = sim.astype(np.float32, copy=False)

    # Base scores for tie-breaking / filling
    base_scores = sim.mean(axis=1)

    # Determine adjacency
    if args.top_l is not None:
        print(f">> Building bipartite graph via top-L per ref: L={args.top_l}")
        adj_by_ref = _build_adj_lists_topl(sim, int(args.top_l))
        tau = None
    else:
        if args.threshold is None:
            # pick threshold by quantile (sampling for speed)
            density = float(args.target_density)
            density = min(max(density, 1e-6), 1.0)
            flat = sim.ravel()
            if flat.size > 2_000_000:
                rng = np.random.default_rng(42)
                flat = rng.choice(flat, size=2_000_000, replace=False)
            tau = float(np.quantile(flat, 1.0 - density))
            print(f">> threshold not provided, using density={density:.4f} => tau~{tau:.4f}")
        else:
            tau = float(args.threshold)
            print(f">> Using provided threshold tau={tau:.4f}")

        adj_by_ref = _build_adj_lists_threshold(
            sim,
            threshold=tau,
            ensure_ref_coverage=bool(args.ensure_ref_coverage),
        )

    n_edges = int(sum(len(x) for x in adj_by_ref))
    density = float(n_edges) / float(n_train * n_ref)
    print(f">> Graph edges={n_edges}  density={density:.6f}")

    # Provide a fast sim_row_getter for the greedy loop
    if args.top_l is not None:
        # when using top-L edges, we still define coverage by membership in top-L
        top_sets = [set(map(int, x.tolist())) for x in adj_by_ref]

        def sim_row_getter(i: int) -> np.ndarray:
            # bool mask for refs that i connects to
            return np.array([i in top_sets[j] for j in range(n_ref)], dtype=np.bool_)

    else:
        assert tau is not None

        def sim_row_getter(i: int) -> np.ndarray:
            return (sim[i] >= tau)

    # Greedy selection (binary coverage)
    greedy_seq, covered = greedy_max_coverage_from_adj(
        adj_by_ref=adj_by_ref,
        sim_row_getter=sim_row_getter,
        n_train=n_train,
        k_max=k_max,
        n_ref=n_ref,
    )
    print(f">> Greedy selected {len(greedy_seq)} items before fill. coverage={covered.mean():.3f}")

    # Fill to k_max by Rep-Sim among remaining
    if len(greedy_seq) < k_max:
        selected_mask = np.zeros(n_train, dtype=np.bool_)
        selected_mask[np.array(greedy_seq, dtype=np.int64)] = True
        remain = np.flatnonzero(~selected_mask)
        need = k_max - len(greedy_seq)
        if need > 0:
            # choose top by base_scores
            cand_scores = base_scores[remain]
            idx = np.argpartition(cand_scores, -need)[-need:]
            fill = remain[idx]
            fill = fill[np.argsort(-cand_scores[idx])]
            greedy_seq.extend(fill.astype(int).tolist())

    # Build scores that encode the greedy ordering
    scores = build_scores_from_sequence(base_scores=base_scores, sequence=greedy_seq, k_max=k_max)

    # Save
    out_path = Path(args.out)
    if out_path.suffix == ".npy":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, scores.astype(np.float32))
        metrics_path = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_path = out_path / "metrics.npy"
        np.save(metrics_path, scores.astype(np.float32))

    print(f">> Saved scores to: {metrics_path}")

    if args.save_selected is not None:
        sel_path = Path(args.save_selected)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        sel_path.write_text(json.dumps(greedy_seq, indent=2))
        print(f">> Saved selected indices to: {sel_path}")


if __name__ == "__main__":
    main()
