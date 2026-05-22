#!/usr/bin/env python
"""Bipartite Greedy Coverage method for DATE-LM fine-tuning data selection.

This method builds a bipartite graph between train and validation embeddings
using a similarity threshold, then runs lazy greedy max-coverage to select
training examples that best cover the validation set.

Usage:
    python probe_bipartite_greedy_instruct.py \
        --train_data_dir /path/to/tulu3.jsonl \
        --val_dataset_name mmlu \
        --val_num_samples 100 \
        --subset_size 200000 \
        --out_dir /path/to/output/metrics.npy \
        --embedding_model BAAI/bge-large-en-v1.5 \
        --device cuda

Alternatively, if you already have precomputed embeddings:
    python probe_bipartite_greedy_instruct.py \
        --train_emb_path /path/to/train_emb.npy \
        --val_emb_path /path/to/val_emb.npy \
        --out_dir /path/to/output/metrics.npy
"""

from __future__ import annotations

import os
import sys
import heapq
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Lazy Greedy Max-Coverage (from SequentialDataVal)
# ============================================================================

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_bytes_1d(x: np.ndarray) -> int:
    return int(_POPCOUNT_LUT[x.astype(np.uint8, copy=False)].sum())


def _pack_bool_matrix(adj: np.ndarray) -> np.ndarray:
    if adj.dtype != np.bool_:
        adj = adj.astype(np.bool_, copy=False)
    return np.packbits(adj, axis=1, bitorder="little")


def lazy_greedy_max_coverage_packed(
    packed_neighbors: np.ndarray, n_ref: int, k: int
) -> List[int]:
    """Lazy greedy max-coverage over packed uint8 bitsets.

    Returns exactly k indices (deterministic fill for the rest).
    """
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
    heap: List[Tuple[int, int]] = [(-int(g), int(i)) for i, g in enumerate(gains0)]
    heapq.heapify(heap)

    chosen = np.zeros(n_train, dtype=np.bool_)
    selected: List[int] = []

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


def coverage_from_packed(
    packed_neighbors: np.ndarray, selected: List[int], n_ref: int
) -> float:
    n_bytes = packed_neighbors.shape[1]
    covered = np.zeros(n_bytes, dtype=np.uint8)
    for idx in selected:
        covered |= packed_neighbors[int(idx)]
    covered_bits = _popcount_bytes_1d(covered)
    return float(covered_bits) / float(n_ref)


# ============================================================================
# Embedding computation
# ============================================================================


def compute_embeddings_st(
    texts: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "cuda",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Compute embeddings using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def compute_embeddings_llm(
    texts: List[str],
    model_path: str,
    tokenizer_name: str,
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 2048,
) -> np.ndarray:
    """Compute embeddings using LLM hidden states (last token)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device
    )
    model.eval()

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing LLM embeddings"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get last layer hidden states
            hidden_states = outputs.hidden_states[-1]
            # Get last token representation (before padding)
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            batch_size_actual = hidden_states.size(0)
            last_token_reps = hidden_states[
                torch.arange(batch_size_actual), seq_lengths
            ]
            all_embeddings.append(last_token_reps.cpu().float().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ============================================================================
# Similarity and threshold learning
# ============================================================================


def compute_similarity_batched(
    train_emb: np.ndarray,
    val_emb: np.ndarray,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    """Compute cosine similarity in batches (GPU accelerated)."""
    try:
        import torch.nn.functional as F

        use_torch = True
    except Exception:
        use_torch = False

    if not use_torch or device == "cpu":
        return train_emb @ val_emb.T

    dev = torch.device(device)
    val_t = torch.from_numpy(val_emb).to(dev)
    val_t = F.normalize(val_t, dim=1)

    out = np.zeros((train_emb.shape[0], val_emb.shape[0]), dtype=np.float32)

    for start in range(0, train_emb.shape[0], batch_size):
        end = min(start + batch_size, train_emb.shape[0])
        batch = torch.from_numpy(train_emb[start:end].astype(np.float32)).to(dev)
        batch = F.normalize(batch, dim=1)
        sim = batch @ val_t.T
        out[start:end] = sim.detach().cpu().numpy()

    return out


def learn_threshold_pilot(
    sim: np.ndarray,
    k: int,
    pilot_size: int = 10000,
    pilot_k: int = 500,
    tau_grid: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[float, dict]:
    """Learn similarity threshold using pilot greedy."""
    if tau_grid is None:
        tau_grid = np.linspace(0.30, 0.80, 21)

    rng = np.random.default_rng(seed)
    n_train, n_ref = sim.shape
    pilot_size = min(pilot_size, n_train)
    pilot_k = min(pilot_k, pilot_size, max(10, k // 50))

    idx = rng.choice(n_train, size=pilot_size, replace=False)
    sim_pilot = sim[idx]

    best_tau = float(tau_grid[0])
    best_obj = -1e9
    best_diag = {}

    for tau in tau_grid:
        adj = sim_pilot >= float(tau)
        packed = _pack_bool_matrix(adj)
        sel = lazy_greedy_max_coverage_packed(packed, n_ref=n_ref, k=pilot_k)
        cov = coverage_from_packed(packed, sel, n_ref=n_ref)
        density = float(adj.mean())

        target = 0.05
        penalty = abs(np.log((density + 1e-9) / target))
        obj = cov - 0.15 * penalty

        if obj > best_obj:
            best_obj = obj
            best_tau = float(tau)
            best_diag = {
                "pilot_coverage": float(cov),
                "graph_density": float(density),
                "objective": float(obj),
            }

    return best_tau, best_diag


# ============================================================================
# DATE-LM compatible data loading
# ============================================================================


def load_train_texts(
    train_data_path: str, subset_size: int = 200000
) -> List[str]:
    """Load training texts from JSONL or HuggingFace dataset."""
    import json

    texts = []
    if train_data_path.endswith(".jsonl"):
        with open(train_data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= subset_size:
                    break
                item = json.loads(line)
                # Handle different formats
                if "messages" in item:
                    # Chat format
                    text = " ".join(
                        m.get("content", "") for m in item["messages"]
                    )
                elif "text" in item:
                    text = item["text"]
                elif "input" in item and "output" in item:
                    text = item["input"] + " " + item["output"]
                else:
                    text = str(item)
                texts.append(text)
    else:
        from datasets import load_dataset

        ds = load_dataset(train_data_path, split="train")
        for i, item in enumerate(ds):
            if i >= subset_size:
                break
            if "text" in item:
                texts.append(item["text"])
            elif "messages" in item:
                text = " ".join(m.get("content", "") for m in item["messages"])
                texts.append(text)

    print(f"Loaded {len(texts)} training examples")
    return texts


def load_val_texts(
    val_dataset_name: str, val_num_samples: int = 100, seed: int = 42
) -> List[str]:
    """Load validation texts from DATE-LM style datasets."""
    # Try to import DATE-LM's dataset utilities
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from minimal_multitask.data import DATASETS
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        if val_dataset_name in DATASETS:
            dataset = DATASETS[val_dataset_name](tokenizer).get_all_test_prompts(
                num_samples=val_num_samples, seed=seed, prompt_only=False
            )
            texts = [item["text"] if "text" in item else str(item) for item in dataset]
            print(f"Loaded {len(texts)} validation examples from {val_dataset_name}")
            return texts
    except Exception as e:
        print(f"Could not load DATE-LM dataset: {e}")

    # Fallback: load from HuggingFace
    from datasets import load_dataset

    dataset_map = {
        "mmlu": ("cais/mmlu", "all", "test"),
        "gsm8k": ("gsm8k", "main", "test"),
        "bbh": ("lukaemon/bbh", None, "test"),
    }

    if val_dataset_name in dataset_map:
        name, subset, split = dataset_map[val_dataset_name]
        if subset:
            ds = load_dataset(name, subset, split=split)
        else:
            ds = load_dataset(name, split=split)

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(ds), size=min(val_num_samples, len(ds)), replace=False)
        texts = []
        for i in indices:
            item = ds[int(i)]
            if "question" in item:
                texts.append(item["question"])
            elif "input" in item:
                texts.append(item["input"])
            else:
                texts.append(str(item))
        print(f"Loaded {len(texts)} validation examples from {val_dataset_name}")
        return texts

    raise ValueError(f"Unknown validation dataset: {val_dataset_name}")


# ============================================================================
# Main
# ============================================================================


def setup(
    # Data paths
    train_data_dir: Optional[str] = None,
    train_emb_path: Optional[str] = None,
    val_emb_path: Optional[str] = None,
    val_dataset_name: str = "mmlu",
    val_num_samples: int = 100,
    subset_size: int = 200000,
    out_dir: str = "out/bipartite_greedy/metrics.npy",
    # Embedding options
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    use_llm_embeddings: bool = False,
    llm_model_path: Optional[str] = None,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    # Selection options
    threshold: Optional[float] = None,
    learn_threshold: bool = True,
    selection_size: Optional[int] = None,  # If None, compute full ranking
    # Hardware
    device: str = "cuda",
    batch_size: int = 64,
    seed: int = 42,
) -> None:
    """Run bipartite greedy coverage data selection.

    This produces a metrics.npy file compatible with DATE-LM's select_data.py.
    """
    out_path = Path(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load or compute embeddings ----
    if train_emb_path and val_emb_path:
        print(f"Loading precomputed embeddings from {train_emb_path}, {val_emb_path}")
        train_emb = np.load(train_emb_path)
        val_emb = np.load(val_emb_path)
    else:
        print("Computing embeddings...")
        train_texts = load_train_texts(train_data_dir, subset_size)
        val_texts = load_val_texts(val_dataset_name, val_num_samples, seed)

        if use_llm_embeddings and llm_model_path:
            train_emb = compute_embeddings_llm(
                train_texts, llm_model_path, tokenizer_name, device, batch_size=8
            )
            val_emb = compute_embeddings_llm(
                val_texts, llm_model_path, tokenizer_name, device, batch_size=8
            )
        else:
            train_emb = compute_embeddings_st(
                train_texts, embedding_model, device, batch_size
            )
            val_emb = compute_embeddings_st(
                val_texts, embedding_model, device, batch_size
            )

        # Optionally save embeddings
        emb_dir = out_path.parent / "embeddings"
        emb_dir.mkdir(exist_ok=True)
        np.save(emb_dir / "train_emb.npy", train_emb)
        np.save(emb_dir / "val_emb.npy", val_emb)
        print(f"Saved embeddings to {emb_dir}")

    n_train, n_ref = train_emb.shape[0], val_emb.shape[0]
    print(f"Train: {n_train}, Val: {n_ref}, Dim: {train_emb.shape[1]}")

    # ---- Compute similarity ----
    print("Computing similarity matrix...")
    sim = compute_similarity_batched(train_emb, val_emb, device, batch_size=8192)

    # ---- Learn threshold ----
    k = selection_size if selection_size else n_train
    if threshold is None:
        if learn_threshold:
            print("Learning threshold via pilot greedy...")
            threshold, diag = learn_threshold_pilot(sim, k=k, seed=seed)
            print(f"Learned threshold: {threshold:.4f}, diag: {diag}")
        else:
            threshold = 0.5  # default

    # ---- Build adjacency and run greedy ----
    print(f"Building bipartite graph with threshold={threshold:.4f}")
    adj = sim >= threshold
    density = float(adj.mean())
    print(f"Graph density: {density:.6f}")

    packed = _pack_bool_matrix(adj)
    print(f"Running lazy greedy max-coverage for k={k}...")
    selected = lazy_greedy_max_coverage_packed(packed, n_ref=n_ref, k=k)
    cov = coverage_from_packed(packed, selected, n_ref)
    print(f"Coverage: {cov:.4f}")

    # ---- Produce scores (DATE-LM compatible) ----
    # Higher score = earlier in greedy order = more valuable
    scores = np.zeros(n_train, dtype=np.float32)
    for rank, idx in enumerate(selected):
        scores[idx] = float(n_train - rank)

    # Normalize to [0, 1] for consistency
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    # ---- Save ----
    np.save(out_path, scores)
    print(f"Saved metrics to {out_path}")

    # Save metadata
    import json

    meta = {
        "method": "bipartite_greedy_coverage",
        "n_train": int(n_train),
        "n_ref": int(n_ref),
        "threshold": float(threshold),
        "graph_density": float(density),
        "coverage": float(cov),
        "selection_size": int(k),
        "embedding_model": embedding_model if not use_llm_embeddings else llm_model_path,
    }
    meta_path = out_path.parent / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse import CLI

    CLI(setup)
