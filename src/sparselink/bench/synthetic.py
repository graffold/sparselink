"""Synthetic network and data generation for benchmarking."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def generate_network(
    n_nodes: int,
    topology: str = "random",
    sparsity: float = 0.2,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a synthetic adjacency matrix.

    Args:
        n_nodes: Number of nodes.
        topology: 'random', 'scalefree', or 'smallworld'.
        sparsity: Off-diagonal edge density (0-1).
        seed: Random seed.

    Returns:
        (n_nodes x n_nodes) adjacency matrix.
    """
    rng = np.random.default_rng(seed)
    if topology == "scalefree":
        A = _scalefree(n_nodes, sparsity, rng)
    elif topology == "smallworld":
        A = _smallworld(n_nodes, sparsity, rng)
    else:
        A = _random_network(n_nodes, sparsity, rng)
    return _stabilize(A)


def _stabilize(A: np.ndarray) -> np.ndarray:
    """Scale network so spectral radius stays below 1."""
    np.fill_diagonal(A, 0.0)
    rho = np.max(np.abs(np.linalg.eigvals(A))) if np.any(A) else 0.0
    if rho > 0.8:
        A = A * (0.8 / rho)
    return A


def _random_network(
    n: int, sparsity: float, rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """Directed random network (Erdos-Renyi)."""
    mask = rng.random((n, n)) < sparsity
    np.fill_diagonal(mask, False)
    weights = rng.standard_normal((n, n))
    return np.where(mask, weights, 0.0).astype(np.float64)


def _scalefree(
    n: int, sparsity: float, rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """Directed scale-free via preferential attachment."""
    m = max(1, int(sparsity * n / 2))
    A = np.zeros((n, n))
    for i in range(min(m + 1, n)):
        for j in range(min(m + 1, n)):
            if i != j:
                A[i, j] = rng.standard_normal()
    for new_node in range(m + 1, n):
        degrees = np.sum(np.abs(A[:new_node]) > 0, axis=1).astype(float)
        total = degrees.sum()
        if total == 0:
            probs = np.ones(new_node) / new_node
        else:
            probs = degrees / total
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            if rng.random() < 0.5:
                A[new_node, t] = rng.standard_normal()
            else:
                A[t, new_node] = rng.standard_normal()
    return A


def _smallworld(
    n: int, sparsity: float, rng: np.random.Generator,
) -> npt.NDArray[np.floating]:
    """Directed Watts-Strogatz small-world network."""
    k = max(2, int(sparsity * n))
    k = k if k % 2 == 0 else k + 1
    beta = 0.3
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k // 2 + 1):
            fwd = (i + j) % n
            bwd = (i - j) % n
            A[i, fwd] = rng.standard_normal()
            A[i, bwd] = rng.standard_normal()
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < beta:
                target = (i + j) % n
                A[i, target] = 0.0
                new_target = rng.integers(0, n)
                while new_target == i or A[i, new_target] != 0:
                    new_target = rng.integers(0, n)
                A[i, new_target] = rng.standard_normal()
    return A


def generate_data(
    network: npt.NDArray[np.floating],
    n_samples: int = 100,
    noise_std: float = 0.1,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate synthetic tabular data from a network.

    Model: X = noise @ (I - A)^{-1}, producing correlated features
    whose dependencies reflect the network structure.

    Args:
        network: (n x n) adjacency matrix.
        n_samples: Number of samples to generate.
        noise_std: Standard deviation of input noise.
        seed: Random seed.

    Returns:
        (n_samples x n_nodes) data matrix.
    """
    rng = np.random.default_rng(seed)
    n = network.shape[0]
    G = np.linalg.inv(np.eye(n) - network)
    noise = rng.normal(0, noise_std, (n_samples, n))
    return noise @ G.T
