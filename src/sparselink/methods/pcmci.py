"""PCMCI causal discovery method."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class PCMCIMethod(InferenceMethod):
    """PCMCI-style causal discovery via lagged partial correlations.

    Implements a simplified PCMCI approach: computes lagged correlations
    then conditions on other variables to remove spurious links.
    """

    name = "pcmci"

    def __init__(self, max_lag: int = 1, threshold: float = 0.0, **kwargs: Any) -> None:
        super().__init__(max_lag=max_lag, threshold=threshold, **kwargs)
        self.max_lag = max_lag
        self.threshold = threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Run PCMCI on time-series data.

        Args:
            X: Data matrix (time_steps x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        T, n = X_arr.shape
        adj = np.zeros((n, n))

        for lag in range(1, self.max_lag + 1):
            if lag >= T:
                break
            X_past = X_arr[: T - lag]
            X_future = X_arr[lag:]
            # Partial correlation between future_j and past_i conditioned on past
            cov_full = np.cov(np.hstack([X_future, X_past]), rowvar=False)
            try:
                prec = np.linalg.inv(cov_full)
            except np.linalg.LinAlgError:
                prec = np.linalg.pinv(cov_full)

            d = np.sqrt(np.diag(prec))
            d[d == 0] = 1.0
            pcor = -prec / np.outer(d, d)

            # Extract cross-block: future (0:n) vs past (n:2n)
            for i in range(n):
                for j in range(n):
                    val = abs(pcor[j, n + i])
                    if val > abs(adj[i, j]):
                        adj[i, j] = val

        adj[adj < self.threshold] = 0.0
        np.fill_diagonal(adj, 0.0)
        return InferenceResult(adjacency_matrix=adj)
