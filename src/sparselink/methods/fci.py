"""FCI algorithm for causal discovery with latent confounders (requires causal-learn)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class FCIMethod(InferenceMethod):
    """FCI algorithm for causal discovery allowing latent confounders.

    Requires: pip install sparselink[causal]
    """

    name = "fci"

    def __init__(self, alpha: float = 0.05, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Run FCI algorithm.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            from causallearn.utils.cit import fisherz
        except ImportError as e:
            raise ImportError(
                "FCI algorithm requires causal-learn. Install with: "
                "pip install sparselink[causal]"
            ) from e

        X_arr = self._to_array(X)
        g, _ = fci(X_arr, fisherz, self.alpha)
        adj = np.abs(np.asarray(g.graph, dtype=np.float64))
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0.0)
        return InferenceResult(adjacency_matrix=adj)
