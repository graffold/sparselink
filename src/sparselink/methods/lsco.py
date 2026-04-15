"""Least Squares with Cutoff (LSCO) network inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class LSCOMethod(InferenceMethod):
    """Infer network via least squares with hard thresholding."""

    name = "lsco"

    def __init__(self, threshold: float = 0.0, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.threshold = threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit least squares and threshold small coefficients.

        Args:
            X: Data matrix (samples x features).
            y: Optional perturbation matrix. If None, uses pseudo-inverse of X.
        """
        X_arr = self._to_array(X)

        if y is not None:
            Y_arr = self._to_array(y)
            # A = -Y * pinv(X) (matching pyGS: A = -P @ pinv(Y))
            A = -Y_arr.T @ np.linalg.pinv(X_arr.T)
        else:
            # Self-regression via pseudo-inverse
            A = np.linalg.pinv(X_arr) @ X_arr
            np.fill_diagonal(A, 0.0)

        # Apply hard threshold
        if self.threshold > 0:
            A[np.abs(A) <= self.threshold] = 0.0

        return InferenceResult(adjacency_matrix=A)
