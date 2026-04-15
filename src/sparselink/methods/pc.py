"""PC algorithm for causal discovery (requires causal-learn)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class PCMethod(InferenceMethod):
    """PC algorithm for constraint-based causal discovery.

    Requires: pip install sparselink[causal]
    """

    name = "pc"

    def __init__(self, alpha: float = 0.05, **kwargs: Any) -> None:
        super().__init__(alpha=alpha, **kwargs)
        self.alpha = alpha

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Run PC algorithm.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc
        except ImportError as e:
            raise ImportError(
                "PC algorithm requires causal-learn. Install with: "
                "pip install sparselink[causal]"
            ) from e

        X_arr = self._to_array(X)
        cg = pc(X_arr, alpha=self.alpha, indep_test="fisherz")
        adj = np.abs(np.asarray(cg.G.graph, dtype=np.float64))
        # Symmetrize: take max of (i,j) and (j,i)
        adj = np.maximum(adj, adj.T)
        np.fill_diagonal(adj, 0.0)
        return InferenceResult(adjacency_matrix=adj)
