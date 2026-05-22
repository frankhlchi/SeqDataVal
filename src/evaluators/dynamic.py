"""Dynamic-programming evaluator (exact, exponential).

This evaluator computes an (approximately) optimal *selection sequence* by
solving a finite-horizon deterministic MDP over subsets.

For RQ1 we use n=20, which still implies 2^n states. This is expensive but is
meant as an optimality-reference baseline.
"""

from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Iterable

import numpy as np
import torch
from opendataval.dataval.api import DataEvaluator, ModelMixin
from sklearn.base import clone as sk_clone
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader, Dataset


class DynamicProgrammingEvaluator(DataEvaluator, ModelMixin):
    """Exact DP evaluator for small training sets.

    Notes on caching
    ----------------
    - We *do* cache utility U(S) for every subset S (this is the dominant reuse).
    - We avoid caching explicit transition lists for every state, and we do not
      snapshot DP tables per subset-size (both can explode memory).
    - We keep a compact DP cache (value/policy arrays indexed by bitmask).
    """

    def __init__(self, max_subset_size: int = 20, random_state=None):
        super().__init__(random_state=random_state)
        self.max_subset_size = int(max_subset_size)

        # Caches (kept for compatibility with older code versions).
        self.value_cache = None  # set to a float array after fitting
        self.transition_cache: dict[int, object] = {}
        self.dp_cache: dict[str, object] = {}

        self.selection_sequence: list[int] = []

    @staticmethod
    def _as_numpy(x: torch.Tensor | Dataset) -> np.ndarray:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, Dataset):
            batch = next(iter(DataLoader(x, batch_size=len(x))))
            if torch.is_tensor(batch):
                return batch.detach().cpu().numpy()
            if isinstance(batch, (list, tuple)) and len(batch) == 1 and torch.is_tensor(batch[0]):
                return batch[0].detach().cpu().numpy()
            return np.asarray(batch)
        return np.asarray(x)

    @staticmethod
    def _to_labels(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 1:
            return y.astype(int, copy=False)
        if y.ndim >= 2 and y.shape[1] == 1:
            return y.reshape(-1).astype(int, copy=False)
        return np.argmax(y, axis=1).astype(int, copy=False)

    @staticmethod
    def _mask_from_combo(combo: Iterable[int]) -> int:
        mask = 0
        for i in combo:
            mask |= 1 << int(i)
        return mask

    def _prepare_numpy_view(self) -> None:
        # Convert once: DP needs millions of subset-fits, so Python/torch Dataset
        # overhead becomes significant if we re-materialize tensors every time.
        self._X_train = self._as_numpy(self.x_train)
        self._y_train = self._to_labels(self._as_numpy(self.y_train))
        self._X_valid = self._as_numpy(self.x_valid)
        self._y_valid = self._to_labels(self._as_numpy(self.y_valid))

        # DP is only used for classification in this repo.
        self._num_classes = int(getattr(self.pred_model, "num_classes", int(self._y_train.max()) + 1))

        # Fast path only supports sklearn-wrapped models (RQ1 uses sklogreg).
        base = getattr(self.pred_model, "model", None)
        if base is None:
            self._sk_model_template = None
        else:
            self._sk_model_template = base

    def _utility_for_mask(self, mask: int, model_kwargs: dict) -> float:
        util = float(self.value_cache[mask])  # type: ignore[index]
        if not math.isnan(util):
            return util

        if mask == 0:
            self.value_cache[mask] = 0.0  # type: ignore[index]
            return 0.0

        # Indices of selected items.
        indices: list[int] = []
        tmp = int(mask)
        while tmp:
            lsb = tmp & -tmp
            indices.append(lsb.bit_length() - 1)
            tmp ^= lsb

        X_sub = self._X_train[indices]
        y_sub = self._y_train[indices]

        if len(np.unique(y_sub)) < self._num_classes:
            model = DummyClassifier(strategy="most_frequent")
        elif self._sk_model_template is not None:
            model = sk_clone(self._sk_model_template)
            if "n_jobs" in getattr(model, "get_params", lambda: {})():
                try:
                    model.set_params(n_jobs=1)
                except Exception:
                    pass
        else:
            # Fallback: use OpenDataVal wrapper (slower).
            curr_model = self.pred_model.clone()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curr_model.fit(
                    torch.from_numpy(X_sub),
                    torch.from_numpy(np.eye(self._num_classes, dtype=np.float32)[y_sub]),
                    **model_kwargs,
                )
            y_hat = curr_model.predict(torch.from_numpy(self._X_valid))
            acc = float((self.y_valid.argmax(dim=1) == y_hat.argmax(dim=1)).float().mean().item())
            self.value_cache[mask] = acc  # type: ignore[index]
            return acc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sub, y_sub)
        y_pred = model.predict(self._X_valid)
        acc = float(np.mean(y_pred == self._y_valid))

        self.value_cache[mask] = acc  # type: ignore[index]
        return acc

    def train_data_values(self, *args, **kwargs):
        self._prepare_numpy_view()

        n_train = int(self._X_train.shape[0])
        eff_max = min(int(self.max_subset_size), n_train)

        # Guardrails: DP is exponential.
        if n_train > 25 or eff_max > 25:
            raise ValueError(
                "DynamicProgrammingEvaluator is exponential and only intended for small "
                "training sets (suggest n<=20). "
                f"Got n_train={n_train}, max_subset_size={eff_max}."
            )

        num_states = 1 << n_train
        self.value_cache = np.full(num_states, np.nan, dtype=np.float32)
        self.value_cache[0] = 0.0

        V = np.full(num_states, np.nan, dtype=np.float32)
        pi = np.full(num_states, -1, dtype=np.int16)

        # DP from larger to smaller subset sizes.
        for size in range(eff_max, -1, -1):
            num_curr = math.comb(n_train, size)
            print(f"[DP] size={size} states={num_curr} (n_train={n_train})")

            for combo in combinations(range(n_train), size):
                mask = self._mask_from_combo(combo)
                u = self._utility_for_mask(mask, kwargs)

                if size == eff_max:
                    V[mask] = u
                    continue

                best_val = float("-inf")
                best_action = -1
                for i in range(n_train):
                    bit = 1 << i
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    future = float(V[next_mask])
                    if future > best_val:
                        best_val = future
                        best_action = i

                if best_action < 0:
                    best_val = 0.0

                V[mask] = u + best_val
                pi[mask] = best_action

        # Reconstruct the optimal sequence.
        seq: list[int] = []
        mask = 0
        while len(seq) < eff_max:
            a = int(pi[mask])
            if a < 0:
                break
            seq.append(a)
            mask |= 1 << a

        # Add remaining points (random tie-break) if max_subset_size < n_train.
        remaining = list(set(range(n_train)) - set(seq))
        try:
            self.random_state.shuffle(remaining)
        except Exception:
            np.random.shuffle(remaining)
        seq.extend(remaining)

        self.selection_sequence = seq

        # Compact DP cache for debugging/repro.
        self.dp_cache = {"V": V, "pi": pi, "n_train": n_train, "eff_max": eff_max}
        return self

    def evaluate_data_values(self) -> np.ndarray:
        n_train = len(self.selection_sequence)
        data_values = np.zeros(n_train, dtype=float)
        for pos, idx in enumerate(self.selection_sequence):
            data_values[idx] = n_train - pos
        return data_values / float(n_train) if n_train else data_values
