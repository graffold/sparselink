"""Elastic Net and Ridge regression network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class ElasticNetMethod(InferenceMethod):
    """Infer sparse network via Elastic Net (L1 + L2 regularization)."""

    name = "elastic_net"

    def __init__(
        self, alpha: float = 0.01, l1_ratio: float = 0.5, **kwargs: Any
    ) -> None:
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit Elastic Net gene-by-gene.

        Args:
            X: Data matrix (samples x features).
            y: Optional target matrix. If None, uses X.
        """
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        A = np.zeros((n_features, n_features))
        targets = self._to_array(y) if y is not None else X_arr

        for i in range(n_features):
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                max_iter=10000,
            )
            model.fit(X_arr, targets[:, i])
            A[i, :] = model.coef_

        return InferenceResult(adjacency_matrix=A)


@registry.register
class RidgeMethod(InferenceMethod):
    """Infer network via Ridge regression (L2 regularization)."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit Ridge regression gene-by-gene.

        Args:
            X: Data matrix (samples x features).
            y: Optional target matrix. If None, uses X.
        """
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        A = np.zeros((n_features, n_features))
        targets = self._to_array(y) if y is not None else X_arr

        for i in range(n_features):
            model = Ridge(alpha=self.alpha, fit_intercept=False)
            model.fit(X_arr, targets[:, i])
            A[i, :] = model.coef_

        return InferenceResult(adjacency_matrix=A)
