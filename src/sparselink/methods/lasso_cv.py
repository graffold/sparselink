"""LassoCV — cross-validated alpha selection for honest benchmarking."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.linear_model import LassoCV
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class LassoCVMethod(InferenceMethod):
    """Infer network via LASSO with cross-validated alpha (no oracle)."""
    name = "lasso_cv"

    def __init__(self, cv: int = 5, max_iter: int = 10000, **kwargs: Any) -> None:
        super().__init__(cv=cv, max_iter=max_iter, **kwargs)
        self.cv = cv
        self.max_iter = max_iter

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        targets = self._to_array(y) if y is not None else X_arr
        A = np.zeros((n_features, n_features))
        for i in range(n_features):
            model = LassoCV(cv=self.cv, max_iter=self.max_iter, fit_intercept=False)
            model.fit(X_arr, targets[:, i])
            A[i, :] = model.coef_
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=A)
