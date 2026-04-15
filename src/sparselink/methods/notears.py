"""NOTEARS: Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import expm

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class NOTEARSMethod(InferenceMethod):
    """NOTEARS linear structure learning via continuous optimization.

    Solves: min ||X - XW||^2 + lambda1*|W|
    subject to: h(W) = tr(e^(W◦W)) - d = 0 (acyclicity)

    Requires: pip install sparselink[causal]
    """

    name = "notears"

    def __init__(
        self,
        lambda1: float = 0.1,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        w_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            lambda1=lambda1,
            max_iter=max_iter,
            h_tol=h_tol,
            w_threshold=w_threshold,
            **kwargs,
        )
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.w_threshold = w_threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Run NOTEARS.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        n_samples, d = X_arr.shape
        X_arr = X_arr - X_arr.mean(axis=0)

        W = self._solve(X_arr, d, n_samples)
        W[np.abs(W) < self.w_threshold] = 0.0
        np.fill_diagonal(W, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(W))

    def _solve(self, X: np.ndarray, d: int, n: int) -> np.ndarray:
        """Augmented Lagrangian solver."""
        W = np.zeros((d, d))
        rho = 1.0
        alpha = 0.0
        h_prev = np.inf

        for _ in range(self.max_iter):
            W = self._minimize(W, X, rho, alpha, d, n)
            h = self._h(W)
            if h < self.h_tol:
                break
            if h > 0.25 * h_prev:
                rho *= 10.0
            alpha += rho * h
            h_prev = h
        return W

    def _minimize(
        self, W: np.ndarray, X: np.ndarray, rho: float, alpha: float, d: int, n: int
    ) -> np.ndarray:
        """Proximal gradient step."""
        from sparselink.accel import matmul

        lr = 1e-3
        for _ in range(300):
            residual = X - matmul(X, W)
            loss_grad = -(2.0 / n) * matmul(X.T, residual)
            E = expm(W * W)
            h_grad = E.T * W * 2.0
            grad = loss_grad + (rho * self._h(W) + alpha) * h_grad
            W_new = W - lr * grad
            # Soft threshold for L1
            W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * self.lambda1, 0.0)
            np.fill_diagonal(W_new, 0.0)
            W = W_new
        return W

    @staticmethod
    def _h(W: np.ndarray) -> float:
        """Acyclicity constraint: tr(e^(W◦W)) - d."""
        d = W.shape[0]
        return float(np.trace(expm(W * W)) - d)
