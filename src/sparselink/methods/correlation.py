"""Partial correlation network inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class PartialCorrelation(InferenceMethod):
    """Infer network via partial correlation (precision matrix)."""

    name = "partial_correlation"

    def __init__(self, threshold: float = 0.0, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Compute partial correlation matrix from data.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        from sparselink.accel import cov as mlx_cov

        X_arr = self._to_array(X)
        cov = mlx_cov(X_arr)
        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)

        # Convert precision to partial correlations
        d = np.sqrt(np.diag(precision))
        d[d == 0] = 1.0
        pcor = -precision / np.outer(d, d)
        np.fill_diagonal(pcor, 0.0)

        # Apply threshold
        pcor[np.abs(pcor) < self.threshold] = 0.0

        return InferenceResult(adjacency_matrix=pcor)
