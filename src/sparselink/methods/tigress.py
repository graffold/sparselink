"""TIGRESS-style stability selection + LARS network inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Lars, LassoLarsIC

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class TIGRESSMethod(InferenceMethod):
    """Infer network via stability selection with LARS (TIGRESS-style)."""

    name = "tigress"

    def __init__(
        self,
        n_bootstrap: int = 50,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_bootstrap=n_bootstrap, random_state=random_state, **kwargs)
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit stability selection per feature to infer adjacency.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        A = np.zeros((n_features, n_features))

        rng = np.random.RandomState(self.random_state)

        for i in range(n_features):
            mask = np.ones(n_features, dtype=bool)
            mask[i] = False
            predictors = X_arr[:, mask]
            target = X_arr[:, i]

            counts = np.zeros(int(mask.sum()))
            for _ in range(self.n_bootstrap):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                X_b, y_b = predictors[idx], target[idx]
                try:
                    model = LassoLarsIC(criterion="bic", max_iter=500)
                    model.fit(X_b, y_b)
                    counts += (np.abs(model.coef_) > 1e-10).astype(float)
                except Exception:
                    try:
                        model = Lars(
                            n_nonzero_coefs=min(5, int(mask.sum())),
                            fit_intercept=True,
                        )
                        model.fit(X_b, y_b)
                        counts += (np.abs(model.coef_) > 1e-10).astype(float)
                    except Exception:
                        continue

            A[mask, i] = counts / self.n_bootstrap

        return InferenceResult(adjacency_matrix=A)
