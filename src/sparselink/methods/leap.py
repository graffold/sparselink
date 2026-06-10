"""LEAP — Lagged Expression Analysis for Prediction."""
from __future__ import annotations
from typing import Any
import numpy as np
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class LEAPMethod(InferenceMethod):
    """Infer network via maximum lagged correlation."""
    name = "leap"

    def __init__(self, max_lag: int = 1, **kwargs: Any) -> None:
        super().__init__(max_lag=max_lag, **kwargs)
        self.max_lag = max_lag

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        A = np.zeros((n_features, n_features))
        for lag in range(1, self.max_lag + 1):
            if lag >= n_samples:
                break
            X_past = X_arr[:-lag]
            X_future = X_arr[lag:]
            # Correlation between past gene i and future gene j
            corr = np.corrcoef(X_past.T, X_future.T)[:n_features, n_features:]
            corr = np.nan_to_num(corr)
            A = np.maximum(A, np.abs(corr))
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=A)
