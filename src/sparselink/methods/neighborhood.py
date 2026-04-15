"""Neighborhood Selection (Meinshausen-Bühlmann) network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lasso

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class NeighborhoodSelection(InferenceMethod):
    """Infer graph structure via node-wise LASSO (Meinshausen-Bühlmann 2006)."""

    name = "neighborhood_selection"

    def __init__(self, alpha: float = 0.1, rule: str = "and", **kwargs: Any) -> None:
        super().__init__(alpha=alpha, rule=rule, **kwargs)
        self.alpha = alpha
        self.rule = rule  # "and" or "or" for symmetrizing

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        support = np.zeros((n_features, n_features))

        for i in range(n_features):
            others = [j for j in range(n_features) if j != i]
            model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000)
            model.fit(X_arr[:, others], X_arr[:, i])
            for k, j in enumerate(others):
                if model.coef_[k] != 0.0:
                    support[i, j] = 1.0

        # Symmetrize
        if self.rule == "and":
            adj = support * support.T
        else:
            adj = np.maximum(support, support.T)

        return InferenceResult(adjacency_matrix=adj)
