"""Graphical LASSO and GLASSO+StARS network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.covariance import GraphicalLasso, LedoitWolf

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class GraphicalLassoMethod(InferenceMethod):
    """Infer sparse precision matrix via Graphical LASSO."""

    name = "glasso"

    def __init__(self, alpha: float = 0.1, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        try:
            model = GraphicalLasso(alpha=self.alpha, max_iter=500)
            model.fit(X_arr)
            prec = model.precision_
        except FloatingPointError:
            # Ill-conditioned: fall back to Ledoit-Wolf shrunk covariance
            lw = LedoitWolf().fit(X_arr)
            prec = np.linalg.pinv(lw.covariance_)
        np.fill_diagonal(prec, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(prec))


@registry.register
class GLASSOStARS(InferenceMethod):
    """GLASSO with StARS (Stability Approach to Regularization Selection)."""

    name = "glasso_stars"

    def __init__(
        self,
        alpha_range: tuple[float, float] = (0.01, 1.0),
        n_alphas: int = 10,
        n_subsamples: int = 20,
        subsample_ratio: float = 0.8,
        beta_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.alpha_range = alpha_range
        self.n_alphas = n_alphas
        self.n_subsamples = n_subsamples
        self.subsample_ratio = subsample_ratio
        self.beta_threshold = beta_threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        sub_size = max(2, int(n_samples * self.subsample_ratio))
        alphas = np.logspace(
            np.log10(self.alpha_range[1]),
            np.log10(self.alpha_range[0]),
            self.n_alphas,
        )
        rng = np.random.default_rng(42)

        best_alpha = alphas[0]
        for alpha in alphas:
            edges = np.zeros((n_features, n_features))
            for _ in range(self.n_subsamples):
                idx = rng.choice(n_samples, size=sub_size, replace=False)
                try:
                    model = GraphicalLasso(alpha=alpha, max_iter=200)
                    model.fit(X_arr[idx])
                    prec = model.precision_
                    edges += (np.abs(prec) > 1e-5).astype(float)
                except Exception:  # noqa: BLE001
                    continue
            freq = edges / self.n_subsamples
            instability = 2.0 * freq * (1.0 - freq)
            np.fill_diagonal(instability, 0.0)
            beta = instability.mean()
            if beta <= self.beta_threshold:
                best_alpha = alpha
                break

        model = GraphicalLasso(alpha=best_alpha, max_iter=500)
        try:
            model.fit(X_arr)
            prec = model.precision_
        except FloatingPointError:
            lw = LedoitWolf().fit(X_arr)
            prec = np.linalg.pinv(lw.covariance_)
        np.fill_diagonal(prec, 0.0)
        return InferenceResult(
            adjacency_matrix=np.abs(prec),
            metadata={"selected_alpha": float(best_alpha)},
        )
