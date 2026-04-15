"""MLX-accelerated linear algebra primitives for Apple Silicon."""

from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

_MIN_SIZE = 1000  # Only use MLX for matrices larger than this


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiply, using MLX on Apple Silicon when available."""
    if HAS_MLX and A.size > _MIN_SIZE:
        # Fall back to numpy if values exceed float32 range
        if np.any(np.abs(A) > 1e30) or np.any(np.abs(B) > 1e30):
            return A @ B
        a = mx.array(A.astype(np.float32))
        b = mx.array(B.astype(np.float32))
        out = a @ b
        mx.eval(out)
        return np.array(out, dtype=np.float64)
    return A @ B


def gram(X: np.ndarray) -> np.ndarray:
    """Compute X^T @ X, accelerated with MLX."""
    if HAS_MLX and X.size > _MIN_SIZE:
        x = mx.array(X.astype(np.float32))
        out = x.T @ x
        mx.eval(out)
        return np.array(out, dtype=np.float64)
    return X.T @ X


def cov(X: np.ndarray) -> np.ndarray:
    """Covariance matrix, accelerated with MLX."""
    X_c = X - X.mean(axis=0)
    n = X_c.shape[0] - 1
    return gram(X_c) / n
