from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Fast popcount for uint8 arrays (numpy 1.24 lacks vectorized bit_count)
_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_bytes_1d(x: np.ndarray) -> int:
    """Popcount of a 1D uint8 array treated as packed bits."""
    if x.dtype != np.uint8:
        x = x.astype(np.uint8, copy=False)
    return int(_POPCOUNT_LUT[x].sum())


def pack_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Pack a boolean adjacency matrix into uint8 bitsets.

    Returns:
        packed: uint8 array of shape (n_train, ceil(n_ref/8)) with bitorder='little'.
    """
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be 2D")
    if adjacency.dtype != np.bool_:
        adjacency = adjacency.astype(np.bool_, copy=False)
    return np.packbits(adjacency, axis=1, bitorder="little")


def lazy_greedy_max_coverage_packed(
    packed_neighbors: np.ndarray,
    n_ref: int,
    k: int,
) -> List[int]:
    """Lazy greedy max-coverage using packed bitsets.

    This is the same solution as standard greedy for monotone submodular
    max-coverage, but is dramatically faster when n_train is large.

    Args:
        packed_neighbors: uint8 packed bitsets, shape (n_train, n_bytes)
        n_ref: number of reference nodes (bits)
        k: number of training examples to select

    Returns:
        selected indices list (length == k, filled deterministically if greedy
        terminates early due to full coverage / no additional gain).
    """

    import heapq

    if packed_neighbors.ndim != 2:
        raise ValueError("packed_neighbors must be 2D")
    if packed_neighbors.dtype != np.uint8:
        packed_neighbors = packed_neighbors.astype(np.uint8, copy=False)

    n_train, n_bytes = packed_neighbors.shape
    if k > n_train:
        raise ValueError(f"k={k} cannot exceed n_train={n_train}")
    if n_ref <= 0:
        raise ValueError("n_ref must be positive")

    # uncovered bitmask (packed)
    uncovered = np.full(n_bytes, 255, dtype=np.uint8)
    if n_ref % 8 != 0:
        uncovered[-1] = (1 << (n_ref % 8)) - 1

    # initial gains = degree (since uncovered = all)
    gains0 = _POPCOUNT_LUT[packed_neighbors].sum(axis=1).astype(np.int64)

    # max-heap via negative keys
    heap: List[Tuple[int, int]] = [(-int(g), int(i)) for i, g in enumerate(gains0)]
    heapq.heapify(heap)

    selected: List[int] = []
    chosen = np.zeros(n_train, dtype=np.bool_)

    while len(selected) < k and heap:
        # If the best *estimated* gain is 0, no remaining point can improve coverage.
        if -heap[0][0] <= 0:
            break

        neg_est, idx = heapq.heappop(heap)
        if chosen[idx]:
            continue

        # recompute true marginal gain
        gain_true = _popcount_bytes_1d(packed_neighbors[idx] & uncovered)

        # look at next-best estimate (lazy greedy acceptance condition)
        best_est_next = -heap[0][0] if heap else 0

        if gain_true >= best_est_next:
            # accept
            selected.append(idx)
            chosen[idx] = True
            uncovered &= ~packed_neighbors[idx]
            if not uncovered.any():
                # fully covered
                break
        else:
            # update key and reinsert
            heapq.heappush(heap, (-int(gain_true), idx))

    # deterministic fill to produce a full ranking (useful for curves)
    if len(selected) < k:
        remaining_idx = np.flatnonzero(~chosen)
        fill = remaining_idx[: (k - len(selected))].astype(int).tolist()
        selected.extend(fill)

    return selected


def cosine_sim_matrix(train_emb: np.ndarray, ref_emb: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix (assumes rows are L2-normalized)."""
    if train_emb.ndim != 2 or ref_emb.ndim != 2:
        raise ValueError("Expected 2D embeddings.")
    if train_emb.shape[1] != ref_emb.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: train={train_emb.shape}, ref={ref_emb.shape}"
        )
    return train_emb @ ref_emb.T


def random_select(n_train: int, k: int, seed: int = 42) -> List[int]:
    if k > n_train:
        raise ValueError(f"k={k} cannot exceed n_train={n_train}")
    rng = np.random.default_rng(seed)
    return rng.choice(n_train, size=k, replace=False).astype(int).tolist()


def repsim_topk_select(sim: np.ndarray, k: int, aggregator: str = "mean") -> List[int]:
    """DATE-LM style Rep-Sim baseline: top-k by aggregated similarity."""
    if sim.ndim != 2:
        raise ValueError("sim must be 2D (n_train, n_ref)")
    if aggregator == "mean":
        scores = sim.mean(axis=1)
    elif aggregator == "max":
        scores = sim.max(axis=1)
    elif aggregator == "sum":
        scores = sim.sum(axis=1)
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    # argpartition for efficiency; stable ordering not required here
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int).tolist()


def build_bipartite_adjacency(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Binary adjacency matrix A[i, j] = 1(sim(i, j) >= threshold)."""
    # For cosine similarity of normalized embeddings, values are in [-1, 1].
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold={threshold} not in [-1, 1]. "
            "If you are using unnormalized embeddings, normalize first."
        )
    return (sim >= threshold).astype(np.bool_)


def greedy_max_coverage(adjacency: np.ndarray, k: int) -> List[int]:
    """Greedy max-coverage selection.

    For large pools (e.g., 200K train points) the naive O(k*n_train*n_ref)
    implementation is too slow. We therefore implement **lazy greedy** over a
    packed-bit representation, which yields the *same greedy solution* for
    monotone submodular coverage but is much faster in practice.
    """
    if adjacency.ndim != 2:
        raise ValueError("adjacency must be 2D")
    n_train, n_ref = adjacency.shape
    if k > n_train:
        raise ValueError(f"k={k} cannot exceed n_train={n_train}")

    packed = pack_adjacency(adjacency)
    return lazy_greedy_max_coverage_packed(packed, n_ref=n_ref, k=k)


def coverage_ratio(adjacency: np.ndarray, selected: List[int]) -> float:
    if len(selected) == 0:
        return 0.0
    n_ref = adjacency.shape[1]
    covered = np.zeros(n_ref, dtype=np.bool_)
    for i in selected:
        covered |= adjacency[i]
    return float(covered.mean())


@dataclass
class ThresholdSearchResult:
    best_threshold: float
    best_score: float
    diagnostics: Dict[str, float]


def learn_threshold_by_graph_stats(
    sim: np.ndarray,
    k_pilot: int,
    thresholds: Optional[np.ndarray] = None,
    target_density: float = 0.02,
    density_tol: float = 0.02,
) -> ThresholdSearchResult:
    """A CPU-friendly threshold selection heuristic.

    In classic SequentialDataVal, Algorithm 1 learns a threshold by aligning
    surrogate utility (coverage) with true utility U(S) measured by retraining.
    For LLM fine-tuning this retraining loop is expensive, so here we provide a
    *cheap* heuristic that tends to produce useful bipartite graphs:

      - encourages high coverage under greedy selection (pilot k)
      - avoids graphs that are too dense (no discrimination) or too sparse (no edges)

    You can later replace this with a warmup-finetune calibration (DATE-LM warmup).
    """

    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 25)

    n_train, n_ref = sim.shape
    k_pilot = int(min(max(1, k_pilot), n_train))

    best_tau = float(thresholds[0])
    best_obj = -1e18
    best_diag: Dict[str, float] = {}

    for tau in thresholds:
        A = build_bipartite_adjacency(sim, float(tau))
        density = float(A.mean())
        # quick pilot selection
        sel = greedy_max_coverage(A, k=k_pilot)
        cov = coverage_ratio(A, sel)

        # density penalty: prefer density near target_density
        # (soft quadratic penalty)
        penalty = ((density - target_density) / max(density_tol, 1e-6)) ** 2
        obj = cov - 0.1 * penalty

        if obj > best_obj:
            best_obj = obj
            best_tau = float(tau)
            best_diag = {
                "pilot_coverage": cov,
                "graph_density": density,
                "density_penalty": float(penalty),
                "objective": float(obj),
            }

    return ThresholdSearchResult(best_threshold=best_tau, best_score=float(best_obj), diagnostics=best_diag)


def sequence_to_scores(n_train: int, sequence: List[int]) -> np.ndarray:
    """Convert a selection sequence to per-example scores.

    Higher score => earlier in sequence.
    This mirrors the ranking->value conversion in `BipartiteMatchingEvaluator`.
    """
    scores = np.zeros(n_train, dtype=np.float32)
    for pos, idx in enumerate(sequence):
        scores[idx] = float(n_train - pos)
    # normalize to [0, 1]
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return scores
