"""GRISLI — Gene Regulation Inference using time-lagged LASSO."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.linear_model import Lasso
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class GRISLIMethod(InferenceMethod):
    """Infer network via LASSO regression on time-lagged expression."""
    name = "grisli"

    def __init__(self, alpha: float = 0.1, n_lags: int = 1, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, n_lags=n_lags, **kwargs)
        self.alpha = alpha
        self.n_lags = n_lags

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        X_lagged = X_arr[:-self.n_lags]
        Y_target = X_arr[self.n_lags:]
        A = np.zeros((n_features, n_features))
        for j in range(n_features):
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=5000)
            model.fit(X_lagged, Y_target[:, j])
            A[j, :] = model.coef_
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(A))
