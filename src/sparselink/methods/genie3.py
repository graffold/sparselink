"""GENIE3-style Random Forest importance network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class GENIE3Method(InferenceMethod):
    """Infer network via Random Forest feature importance (GENIE3-style)."""

    name = "genie3"

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: str = "sqrt",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit Random Forest per gene to infer importance-based adjacency.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        A = np.zeros((n_features, n_features))

        for i in range(n_features):
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            predictors = X_arr[:, mask]
            target = X_arr[:, i]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                random_state=self.random_state,
            )
            rf.fit(predictors, target)
            A[mask, i] = rf.feature_importances_

        return InferenceResult(adjacency_matrix=A)
