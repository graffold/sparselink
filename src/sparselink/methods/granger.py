"""Granger Causality network inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class GrangerCausality(InferenceMethod):
    """Pairwise Granger Causality via F-test on VAR models.

    Tests whether lagged values of variable i improve prediction of variable j
    beyond j's own lags.
    """

    name = "granger_causality"

    def __init__(self, max_lag: int = 1, threshold: float = 0.0, **kwargs: Any) -> None:
        super().__init__(max_lag=max_lag, threshold=threshold, **kwargs)
        self.max_lag = max_lag
        self.threshold = threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Compute pairwise Granger causality F-statistics.

        Args:
            X: Time-series data (time_steps x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        T, n = X_arr.shape
        adj = np.zeros((n, n))

        for j in range(n):
            # Build lagged matrix for restricted model (only own lags)
            Y = X_arr[self.max_lag :, j]
            n_obs = len(Y)
            Z_restricted = np.column_stack(
                [
                    X_arr[self.max_lag - lag : T - lag, j]
                    for lag in range(1, self.max_lag + 1)
                ]
            )
            # Restricted RSS
            rss_r = self._ols_rss(Z_restricted, Y)

            for i in range(n):
                if i == j:
                    continue
                # Unrestricted: add lags of variable i
                Z_unrestricted = np.column_stack(
                    [
                        Z_restricted,
                        *[
                            X_arr[self.max_lag - lag : T - lag, i : i + 1]
                            for lag in range(1, self.max_lag + 1)
                        ],
                    ]
                )
                rss_u = self._ols_rss(Z_unrestricted, Y)
                # F-statistic
                df_diff = self.max_lag
                df_resid = n_obs - Z_unrestricted.shape[1]
                if df_resid > 0 and rss_u > 0:
                    f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)
                    adj[i, j] = max(f_stat, 0.0)

        adj[adj < self.threshold] = 0.0
        return InferenceResult(adjacency_matrix=adj)

    @staticmethod
    def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
        """Compute residual sum of squares from OLS."""
        coef, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        return float(np.sum((y - y_hat) ** 2))
