"""Bayesian structure learning with BDeu and BGe scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import gammaln

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


def _bdeu_local_score(
    data: np.ndarray, child: int, parents: list[int], ess: float, n_bins: int
) -> float:
    """Compute BDeu local score for a child given parents.

    Discretizes continuous data into bins, then computes the BDeu score
    using sufficient statistics (counts).
    """
    n_samples = data.shape[0]
    # Discretize
    child_data = np.digitize(
        data[:, child],
        np.linspace(data[:, child].min(), data[:, child].max(), n_bins + 1)[1:-1],
    )
    if parents:
        parent_data = np.column_stack(
            [
                np.digitize(
                    data[:, p],
                    np.linspace(data[:, p].min(), data[:, p].max(), n_bins + 1)[1:-1],
                )
                for p in parents
            ]
        )
        # Encode parent configurations as single integer
        parent_configs = np.zeros(n_samples, dtype=int)
        multiplier = 1
        for col in range(parent_data.shape[1]):
            parent_configs += parent_data[:, col] * multiplier
            multiplier *= n_bins
        unique_configs = np.unique(parent_configs)
        q_i = len(unique_configs)
    else:
        parent_configs = np.zeros(n_samples, dtype=int)
        unique_configs = np.array([0])
        q_i = 1

    r_i = n_bins  # number of states for child
    alpha_ijk = ess / (q_i * r_i)
    alpha_ij = ess / q_i

    score = 0.0
    for j_val in unique_configs:
        mask_j = parent_configs == j_val
        n_ij = mask_j.sum()
        score += gammaln(alpha_ij) - gammaln(alpha_ij + n_ij)
        for k in range(r_i):
            n_ijk = ((child_data[mask_j]) == k).sum()
            score += gammaln(alpha_ijk + n_ijk) - gammaln(alpha_ijk)

    return float(score)


def _bge_local_score(
    data: np.ndarray, child: int, parents: list[int], alpha_mu: float, alpha_w: float
) -> float:
    """Compute BGe (Bayesian Gaussian equivalent) local score.

    Uses the closed-form marginal likelihood for Gaussian data with
    normal-Wishart prior.
    """
    n = data.shape[0]
    variables = [child] + parents
    p = len(variables)
    sub_data = data[:, variables]

    # Prior parameters
    T0 = np.eye(p) * (alpha_w - p - 1)  # prior scatter matrix
    mu0 = np.zeros(p)

    # Posterior parameters
    mean_data = sub_data.mean(axis=0)
    S = (sub_data - mean_data).T @ (sub_data - mean_data)
    diff = mean_data - mu0
    Tn = T0 + S + (alpha_mu * n / (alpha_mu + n)) * np.outer(diff, diff)

    alpha_n = alpha_w + n

    # Log marginal likelihood (up to constant across models with same child)
    score = 0.0
    score += -0.5 * n * p * np.log(np.pi)
    score += 0.5 * p * (np.log(alpha_mu) - np.log(alpha_mu + n))
    # Multivariate gamma functions via gammaln
    for i in range(p):
        score += gammaln(0.5 * (alpha_n - i)) - gammaln(0.5 * (alpha_w - i))
    score += 0.5 * alpha_w * np.linalg.slogdet(T0)[1]
    score -= 0.5 * alpha_n * np.linalg.slogdet(Tn)[1]

    return float(score)


def _greedy_hill_climbing(
    data: np.ndarray, score_fn: Any, max_parents: int
) -> np.ndarray:
    """Greedy hill-climbing to find DAG structure maximizing score."""
    n_vars = data.shape[1]
    adj = np.zeros((n_vars, n_vars))  # adj[i,j] = 1 means edge i -> j

    # Compute initial scores (no parents)
    current_scores = np.array([score_fn(data, i, []) for i in range(n_vars)])

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_edge = (-1, -1)

        for j in range(n_vars):
            current_parents = list(np.where(adj[:, j] != 0)[0])
            if len(current_parents) >= max_parents:
                continue
            for i in range(n_vars):
                if i == j or adj[i, j] != 0:
                    continue
                # Check acyclicity: adding i->j must not create cycle
                if _would_create_cycle(adj, i, j):
                    continue
                new_parents = current_parents + [i]
                new_score = score_fn(data, j, new_parents)
                gain = new_score - current_scores[j]
                if gain > best_gain:
                    best_gain = gain
                    best_edge = (i, j)

        if best_edge[0] >= 0:
            i, j = best_edge
            adj[i, j] = 1.0
            current_parents = list(np.where(adj[:, j] != 0)[0])
            current_scores[j] = score_fn(data, j, current_parents)
            improved = True

    return adj


def _would_create_cycle(adj: np.ndarray, source: int, target: int) -> bool:
    """Check if adding source->target would create a cycle via DFS."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    stack = [target]
    while stack:
        node = stack.pop()
        if node == source:
            return True
        if visited[node]:
            continue
        visited[node] = True
        children = np.where(adj[node, :] != 0)[0]
        stack.extend(children.tolist())
    return False


@registry.register
class BDeuMethod(InferenceMethod):
    """Bayesian structure learning with BDeu scoring.

    Uses greedy hill-climbing to find the DAG that maximizes the BDeu score.
    """

    name = "bdeu"

    def __init__(
        self, ess: float = 10.0, n_bins: int = 3, max_parents: int = 3, **kwargs: Any
    ) -> None:
        super().__init__(ess=ess, n_bins=n_bins, max_parents=max_parents, **kwargs)
        self.ess = ess
        self.n_bins = n_bins
        self.max_parents = max_parents

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit BDeu structure learning.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)

        def score_fn(data: np.ndarray, child: int, parents: list[int]) -> float:
            return _bdeu_local_score(data, child, parents, self.ess, self.n_bins)

        adj = _greedy_hill_climbing(X_arr, score_fn, self.max_parents)
        return InferenceResult(adjacency_matrix=adj)


@registry.register
class BGeMethod(InferenceMethod):
    """Bayesian structure learning with BGe (Gaussian) scoring.

    Uses greedy hill-climbing to find the DAG that maximizes the BGe score.
    Appropriate for continuous data assumed to be Gaussian.
    """

    name = "bge"

    def __init__(
        self,
        alpha_mu: float = 1.0,
        alpha_w: float | None = None,
        max_parents: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            alpha_mu=alpha_mu, alpha_w=alpha_w, max_parents=max_parents, **kwargs
        )
        self.alpha_mu = alpha_mu
        self._alpha_w = alpha_w
        self.max_parents = max_parents

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Fit BGe structure learning.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        X_arr = self._to_array(X)
        p = X_arr.shape[1]
        # Default alpha_w must be > p + 1 for proper prior
        alpha_w = self._alpha_w if self._alpha_w is not None else float(p + 2)

        def score_fn(data: np.ndarray, child: int, parents: list[int]) -> float:
            return _bge_local_score(data, child, parents, self.alpha_mu, alpha_w)

        adj = _greedy_hill_climbing(X_arr, score_fn, self.max_parents)
        return InferenceResult(adjacency_matrix=adj)
