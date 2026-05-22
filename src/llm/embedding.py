from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np


EmbedMode = Literal["query", "doc"]


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={x.shape}")
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


@dataclass
class BaseEmbedder:
    """Interface for text embedders.

    For CPU-only sandbox runs we default to TF-IDF.
    For real runs (GPU/internet), you can swap in BGE/E5/LLM hidden states.
    """

    name: str

    def encode(self, texts: List[str], mode: EmbedMode) -> np.ndarray:
        raise NotImplementedError


@dataclass
class TfidfEmbedder(BaseEmbedder):
    """A deterministic, CPU-friendly embedder.

    This is NOT meant to be SOTA; it is used to validate the selection
    pipeline end-to-end in environments without model downloads.
    """

    ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 50000
    lowercase: bool = True
    _vectorizer: Optional[object] = None

    def __init__(
        self,
        name: str = "tfidf",
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int = 50000,
        lowercase: bool = True,
    ):
        super().__init__(name=name)
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase
        self._vectorizer = None

    def fit(self, corpus: List[str]) -> "TfidfEmbedder":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for TfidfEmbedder. "
                "Install with `pip install scikit-learn`."
            ) from e

        self._vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=self.lowercase,
            norm=None,  # we normalize ourselves
        )
        self._vectorizer.fit(corpus)
        return self

    def encode(self, texts: List[str], mode: EmbedMode) -> np.ndarray:
        # TF-IDF does not need query/doc prompt separation.
        if self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fit() before encode().")
        x = self._vectorizer.transform(texts)
        x = x.astype(np.float32)
        dense = x.toarray()
        return _l2_normalize(dense)


@dataclass
class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer / BGE / E5 embedder.

    This class is optional; it will only work if:
      1) `sentence-transformers` is installed
      2) the model weights are available (downloaded) in the environment

    In this sandbox we keep it as a drop-in for your GPU environment.
    """

    model_name: str = "BAAI/bge-large-en-v1.5"
    query_prefix: str = "Represent this question for retrieving relevant training data: "
    device: str = "cpu"
    batch_size: int = 64

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        query_prefix: str = "Represent this question for retrieving relevant training data: ",
        device: str = "cpu",
        batch_size: int = 64,
        name: str = "sentence-transformer",
    ):
        super().__init__(name=name)
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "SentenceTransformerEmbedder requires `sentence-transformers`. "
                "Install with `pip install sentence-transformers`."
            ) from e
        self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str], mode: EmbedMode) -> np.ndarray:
        self._lazy_load()
        if mode == "query":
            texts = [self.query_prefix + t for t in texts]
        # sentence-transformers can normalize embeddings internally
        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        return _l2_normalize(emb)
