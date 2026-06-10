"""DSPIN — Discrete SPIn Network inference via pseudolikelihood."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class DSPINMethod(InferenceMethod):
    """Infer network via pseudolikelihood on binarized expression (Ising model)."""
    name = "dspin"

    def __init__(self, alpha: float = 0.1, n_bins: int = 2, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, n_bins=n_bins, **kwargs)
        self.alpha = alpha
        self.n_bins = n_bins

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        X_arr = self._to_array(X)
        n_samples, n_features = X_arr.shape
        # Binarize: above/below median per gene
        X_bin = np.zeros_like(X_arr, dtype=int)
        for i in range(n_features):
            X_bin[:, i] = (X_arr[:, i] > np.median(X_arr[:, i])).astype(int)
        # Pseudolikelihood: for each gene, logistic regression against all others
        A = np.zeros((n_features, n_features))
        C = 1.0 / self.alpha  # sklearn uses inverse regularization
        for j in range(n_features):
            mask = np.ones(n_features, dtype=bool)
            mask[j] = False
            X_others = X_bin[:, mask]
            y_j = X_bin[:, j]
            if len(np.unique(y_j)) < 2:
                continue
            model = LogisticRegression(C=C, penalty='l1', solver='liblinear',
                                       fit_intercept=True, max_iter=500)
            model.fit(X_others, y_j)
            A[j, mask] = model.coef_[0]
        # Symmetrize
        A = (A + A.T) / 2.0
        np.fill_diagonal(A, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(A))
