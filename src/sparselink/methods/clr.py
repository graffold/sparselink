"""Context Likelihood of Relatedness (CLR) network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mutual_info_score

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class CLRMethod(InferenceMethod):
    """Infer network via CLR (mutual information + z-score normalization)."""

    name = "clr"

    def __init__(
        self, threshold: float = 0.0, n_bins: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(threshold=threshold, n_bins=n_bins, **kwargs)
        self.threshold = threshold
        self.n_bins = n_bins

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Compute CLR network from data.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape

        # Discretize for MI calculation
        n_bins = self.n_bins or max(2, min(int(np.sqrt(n_samples)), 10))
        discretized = np.zeros((n_features, n_samples), dtype=int)
        for i in range(n_features):
            ranks = np.argsort(np.argsort(X_arr[:, i]))
            discretized[i, :] = (ranks * n_bins) // n_samples

        # Compute MI matrix
        mi = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                val = mutual_info_score(discretized[i], discretized[j])
                mi[i, j] = val
                mi[j, i] = val

        # CLR z-score transform
        mask = ~np.eye(n_features, dtype=bool)
        means = np.array([mi[i, mask[i]].mean() for i in range(n_features)])
        stds = np.array([mi[i, mask[i]].std() for i in range(n_features)])
        stds[stds == 0] = 1.0

        Z = (mi - means[:, None]) / stds[:, None]
        clr_matrix = np.sqrt(Z**2 + Z.T**2)
        np.fill_diagonal(clr_matrix, 0.0)

        # Apply threshold
        if self.threshold > 0:
            clr_matrix[clr_matrix < self.threshold] = 0.0

        return InferenceResult(adjacency_matrix=clr_matrix)
