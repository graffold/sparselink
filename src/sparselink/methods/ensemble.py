"""Ensemble — Borda-count consensus across multiple methods."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import rankdata
from sparselink.base import InferenceMethod
from sparselink.registry import registry, get_method
from sparselink.types import InferenceResult, InputData


@registry.register
class EnsembleMethod(InferenceMethod):
    """Consensus network via Borda-count rank aggregation."""
    name = "ensemble"

    def __init__(self, method_names: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.method_names = method_names or ["lasso", "genie3", "ridge", "partial_correlation"]

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n = X_arr.shape[1]
        rank_sum = np.zeros((n, n))
        for name in self.method_names:
            result = get_method(name)().fit(X_arr, y)
            weights = np.abs(result.adjacency_matrix).ravel()
            rank_sum += rankdata(weights).reshape(n, n)
        mn, mx = rank_sum.min(), rank_sum.max()
        A = (rank_sum - mn) / (mx - mn) if mx > mn else np.zeros((n, n))
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=A)
