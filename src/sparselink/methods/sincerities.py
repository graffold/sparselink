"""SINCERITIES — perturbation-response network inference."""
from __future__ import annotations
from typing import Any
import numpy as np
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class SINCERITIESMethod(InferenceMethod):
    """Infer network via regularized least squares on perturbation response."""
    name = "sincerities"

    def __init__(self, alpha: float = 0.01, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_features = X_arr.shape[1]
        if y is not None:
            P = self._to_array(y)  # samples x features
            Y = X_arr.T  # features x samples
            Pt = P.T  # features x samples
            PtP = Pt @ Pt.T
            A = Y @ P @ np.linalg.inv(PtP + self.alpha * np.eye(n_features))
        else:
            dX = np.diff(X_arr, axis=0)
            A = np.corrcoef(dX.T)
            A = np.nan_to_num(A)
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(A))
