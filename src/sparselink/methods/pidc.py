"""PIDC — Partial Information Decomposition and Context."""
from __future__ import annotations
from typing import Any
import numpy as np
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class PIDCMethod(InferenceMethod):
    """Infer network via mutual information with PID-based context correction."""
    name = "pidc"

    def __init__(self, n_bins: int = 10, **kwargs: Any) -> None:
        super().__init__(n_bins=n_bins, **kwargs)
        self.n_bins = n_bins

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        # Discretize
        X_disc = np.zeros((n_samples, n_features), dtype=int)
        for i in range(n_features):
            bins = np.percentile(X_arr[:, i], np.linspace(0, 100, self.n_bins + 1)[1:-1])
            X_disc[:, i] = np.clip(np.digitize(X_arr[:, i], bins), 0, self.n_bins - 1)
        # Pairwise MI
        mi = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                val = self._mi(X_disc[:, i], X_disc[:, j])
                mi[i, j] = mi[j, i] = val
        # PUC scores
        eps = 1e-15
        A = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                max_i = np.max(np.delete(mi[i], j))
                max_j = np.max(np.delete(mi[j], i))
                A[i, j] = A[j, i] = mi[i, j] / max(max_i, max_j, eps)
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=A)

    def _mi(self, x: np.ndarray, y: np.ndarray) -> float:
        hist = np.zeros((self.n_bins, self.n_bins))
        for xi, yi in zip(x, y):
            hist[xi, yi] += 1
        pxy = hist / hist.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        mask = pxy > 0
        denom = (px * py)[mask]
        return float(np.sum(pxy[mask] * np.log(pxy[mask] / denom)))
