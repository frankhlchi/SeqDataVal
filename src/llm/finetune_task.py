from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from llm.embedding import BaseEmbedder, TfidfEmbedder
from llm.selectors import (
    ThresholdSearchResult,
    build_bipartite_adjacency,
    cosine_sim_matrix,
    coverage_ratio,
    greedy_max_coverage,
    learn_threshold_by_graph_stats,
    random_select,
    repsim_topk_select,
    sequence_to_scores,
)


SelectionMethod = Literal["random", "repsim", "bipartite"]


@dataclass
class TrainExample:
    """A minimal training example for SFT/finetune selection."""

    instruction: str
    response: str = ""
    meta: Optional[Dict] = None


@dataclass
class RefExample:
    """Reference/validation example representing target capability."""

    question: str
    meta: Optional[Dict] = None


@dataclass
class SelectionOutputs:
    selected_indices: List[int]
    scores: np.ndarray
    metrics: Dict[str, float]
    threshold: Optional[float] = None


class FinetuneDataSelectionTask:
    """A small, CPU-runnable task scaffold for LLM fine-tuning data selection.

    This is NOT training a model. It focuses on the *data selection stage*:
      (train pool, reference set) -> scores / selected indices.

    In your GPU environment, you can plug the produced scores into DATE-LM's
    `methods/select_data.py`, or directly fine-tune on the selected subset.
    """

    def __init__(
        self,
        train: List[TrainExample],
        ref: List[RefExample],
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.train = train
        self.ref = ref
        self.embedder = embedder or TfidfEmbedder()

        self._train_emb: Optional[np.ndarray] = None
        self._ref_emb: Optional[np.ndarray] = None

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_ref(self) -> int:
        return len(self.ref)

    def _build_texts(self) -> Tuple[List[str], List[str]]:
        train_texts = [ex.instruction for ex in self.train]
        ref_texts = [ex.question for ex in self.ref]
        return train_texts, ref_texts

    def compute_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        train_texts, ref_texts = self._build_texts()

        # TF-IDF must be fit on a corpus first.
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(train_texts + ref_texts)

        self._train_emb = self.embedder.encode(train_texts, mode="doc")
        self._ref_emb = self.embedder.encode(ref_texts, mode="query")
        return self._train_emb, self._ref_emb

    def select(
        self,
        k: int,
        method: SelectionMethod = "bipartite",
        threshold: Optional[float] = None,
        learn_threshold: bool = False,
        seed: int = 42,
        repsim_aggregator: str = "mean",
    ) -> SelectionOutputs:
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self.n_train:
            raise ValueError(f"k={k} cannot exceed n_train={self.n_train}")

        if self._train_emb is None or self._ref_emb is None:
            self.compute_embeddings()

        train_emb = self._train_emb
        ref_emb = self._ref_emb
        sim = cosine_sim_matrix(train_emb, ref_emb)

        if method == "random":
            sel = random_select(self.n_train, k=k, seed=seed)
            scores = np.zeros(self.n_train, dtype=np.float32)
            scores[sel] = 1.0
            metrics = {
                "coverage": float("nan"),
                "avg_similarity": float(sim[sel].mean()),
            }
            return SelectionOutputs(selected_indices=sel, scores=scores, metrics=metrics)

        if method == "repsim":
            sel = repsim_topk_select(sim, k=k, aggregator=repsim_aggregator)
            # score = aggregated similarity (normalized)
            if repsim_aggregator == "mean":
                raw = sim.mean(axis=1)
            elif repsim_aggregator == "max":
                raw = sim.max(axis=1)
            elif repsim_aggregator == "sum":
                raw = sim.sum(axis=1)
            else:
                raise ValueError(f"Unknown aggregator: {repsim_aggregator}")
            scores = raw.astype(np.float32)
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            metrics = {
                "coverage": float("nan"),
                "avg_similarity": float(sim[sel].mean()),
            }
            return SelectionOutputs(selected_indices=sel, scores=scores, metrics=metrics)

        if method != "bipartite":
            raise ValueError(f"Unknown method: {method}")

        # bipartite coverage
        tau: Optional[float] = threshold
        threshold_info: Optional[ThresholdSearchResult] = None
        if learn_threshold or tau is None:
            threshold_info = learn_threshold_by_graph_stats(sim, k_pilot=min(500, max(1, k // 2)))
            tau = threshold_info.best_threshold

        assert tau is not None
        A = build_bipartite_adjacency(sim, threshold=float(tau))
        seq = greedy_max_coverage(A, k=k)
        cov = coverage_ratio(A, seq)
        scores = sequence_to_scores(self.n_train, seq)

        metrics: Dict[str, float] = {
            "coverage": float(cov),
            "graph_density": float(A.mean()),
            "avg_similarity": float(sim[seq].mean()),
        }
        if threshold_info is not None:
            for k_, v_ in threshold_info.diagnostics.items():
                metrics[f"threshold_search/{k_}"] = float(v_)

        return SelectionOutputs(
            selected_indices=seq,
            scores=scores,
            metrics=metrics,
            threshold=float(tau),
        )

    def save_outputs(self, out_dir: str | Path, outputs: SelectionOutputs) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "selected_indices.json").write_text(
            json.dumps(outputs.selected_indices, indent=2, ensure_ascii=False)
        )
        # `scores.npy` is our local name; `metrics.npy` matches DATE-LM's expectation.
        np.save(out_dir / "scores.npy", outputs.scores)
        np.save(out_dir / "metrics.npy", outputs.scores)

        meta = {
            "n_train": self.n_train,
            "n_ref": self.n_ref,
            "threshold": outputs.threshold,
            "metrics": outputs.metrics,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
