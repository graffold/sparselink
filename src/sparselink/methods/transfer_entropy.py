"""Transfer Entropy network inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class TransferEntropy(InferenceMethod):
    """Transfer Entropy via kernel density estimation (KNN-based).

    Uses a binning approach to estimate conditional entropies for
    computing transfer entropy between pairs of variables.
    """

    name = "transfer_entropy"

    def __init__(
        self, max_lag: int = 1, n_bins: int = 10, threshold: float = 0.0, **kwargs: Any
    ) -> None:
        super().__init__(max_lag=max_lag, n_bins=n_bins, threshold=threshold, **kwargs)
        self.max_lag = max_lag
        self.n_bins = n_bins
        self.threshold = threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Compute pairwise transfer entropy.

        Args:
            X: Time-series data (time_steps x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        T, n = X_arr.shape
        adj = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                te = self._compute_te(X_arr[:, i], X_arr[:, j], T)
                adj[i, j] = max(te, 0.0)

        adj[adj < self.threshold] = 0.0
        return InferenceResult(adjacency_matrix=adj)

    def _compute_te(self, source: np.ndarray, target: np.ndarray, T: int) -> float:
        """Compute TE from source -> target using binning."""
        lag = self.max_lag
        if lag >= T - 1:
            return 0.0

        # Discretize
        s_binned = self._digitize(source)
        t_binned = self._digitize(target)

        # TE(X->Y) = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
        y_future = t_binned[lag:]
        y_past = t_binned[: T - lag]
        x_past = s_binned[: T - lag]

        h_y_given_ypast = self._cond_entropy(y_future, y_past)
        # Joint conditioning on (y_past, x_past)
        joint_past = y_past * self.n_bins + x_past
        h_y_given_joint = self._cond_entropy(y_future, joint_past)

        return h_y_given_ypast - h_y_given_joint

    def _digitize(self, x: np.ndarray) -> np.ndarray:
        """Bin continuous values into discrete bins."""
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.zeros(len(x), dtype=int)
        bins = np.linspace(x_min, x_max, self.n_bins + 1)
        return np.clip(np.digitize(x, bins[1:-1]), 0, self.n_bins - 1)

    @staticmethod
    def _cond_entropy(y: np.ndarray, cond: np.ndarray) -> float:
        """H(Y|C) via counting."""
        n = len(y)
        if n == 0:
            return 0.0
        # Joint counts
        joint = y * (cond.max() + 1) + cond
        _, joint_counts = np.unique(joint, return_counts=True)
        _, cond_counts = np.unique(cond, return_counts=True)

        h_joint = -np.sum(joint_counts / n * np.log2(joint_counts / n + 1e-12))
        h_cond = -np.sum(cond_counts / n * np.log2(cond_counts / n + 1e-12))
        return float(h_joint - h_cond)
