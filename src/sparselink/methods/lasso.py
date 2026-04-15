"""LASSO-based network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class LassoMethod(InferenceMethod):
    """Infer sparse network via LASSO regression (gene-by-gene)."""

    name = "lasso"

    def __init__(self, alpha: float = 0.01, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit LASSO to infer adjacency matrix.

        Args:
            X: Data matrix (samples x features).
            y: Optional perturbation matrix. If None, uses X as both features and targets.
        """
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        A = np.zeros((n_features, n_features))

        targets = self._to_array(y) if y is not None else X_arr

        for i in range(n_features):
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10000)
            model.fit(X_arr, targets[:, i])
            A[i, :] = model.coef_

        return InferenceResult(adjacency_matrix=A)
