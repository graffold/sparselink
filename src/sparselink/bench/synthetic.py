"""Synthetic network and expression data generation (GeneSpider-compatible)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def generate_network(
    n_genes: int,
    topology: str = "random",
    sparsity: float = 0.2,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate a synthetic adjacency matrix matching GeneSpider conventions.

    Key properties (matching GeneSpider):
    - Directed (asymmetric)
    - Negative diagonal (self-regulation, ensures stability)
    - Sparse off-diagonal edges

    Args:
        n_genes: Number of nodes.
        topology: 'random', 'scalefree', or 'smallworld'.
        sparsity: Off-diagonal edge density (0-1). GeneSpider typical: ~0.04-0.1.
        seed: Random seed.

    Returns:
        (n_genes x n_genes) adjacency matrix with negative diagonal.
    """
    rng = np.random.default_rng(seed)

    if topology == "scalefree":
        A = _scalefree(n_genes, sparsity, rng)
    elif topology == "smallworld":
        A = _smallworld(n_genes, sparsity, rng)
    else:
        A = _random_network(n_genes, sparsity, rng)

    return _stabilize(A, rng)


def _stabilize(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stabilize network GeneSpider-style: negative diagonal + scale."""
    n = A.shape[0]
    # Set diagonal to negative self-regulation (like GeneSpider)
    np.fill_diagonal(A, 0.0)
    # Scale off-diagonal so spectral radius of (I-A) is well-conditioned
    off_diag = A.copy()
    np.fill_diagonal(off_diag, 0.0)
    rho = np.max(np.abs(np.linalg.eigvals(off_diag))) if np.any(off_diag) else 0.0
    if rho > 0.8:
        A = off_diag * (0.8 / rho)
    # Add negative diagonal (self-degradation, typical range -0.1 to -1.6)
    diag_vals = -(0.1 + 1.5 * rng.random(n))
    np.fill_diagonal(A, diag_vals)
    return A


def _random_network(
    n: int, sparsity: float, rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    """Directed random network (Erdős–Rényi)."""
    mask = rng.random((n, n)) < sparsity
    np.fill_diagonal(mask, False)
    weights = rng.standard_normal((n, n))
    return np.where(mask, weights, 0.0).astype(np.float64)


def _scalefree(
    n: int, sparsity: float, rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    """Directed scale-free via preferential attachment (GeneSpider-style)."""
    m = max(1, int(sparsity * n / 2))
    A = np.zeros((n, n))
    # Seed: fully connected directed core
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
            # Directed: randomly choose in or out edge
            if rng.random() < 0.5:
                A[new_node, t] = rng.standard_normal()
            else:
                A[t, new_node] = rng.standard_normal()
    return A


def _smallworld(
    n: int, sparsity: float, rng: np.random.Generator
) -> npt.NDArray[np.floating]:
    """Directed Watts-Strogatz small-world network."""
    k = max(2, int(sparsity * n))
    k = k if k % 2 == 0 else k + 1
    beta = 0.3
    A = np.zeros((n, n))
    # Directed ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            fwd = (i + j) % n
            bwd = (i - j) % n
            A[i, fwd] = rng.standard_normal()
            A[i, bwd] = rng.standard_normal()
    # Rewire
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


def generate_expression(
    network: npt.NDArray[np.floating],
    n_samples: int = 100,
    snr: float = 10.0,
    seed: int | None = None,
) -> npt.NDArray[np.floating]:
    """Generate synthetic expression data matching GeneSpider protocol.

    Model: Y = G @ P + E, where G = inv(I - A).
    P is a sparse perturbation matrix (identity-like, one perturbation per experiment).
    SNR follows GeneSpider: stdE = s1(G@P) / (SNR * sqrt(chi2(1-alpha, |P|))).

    Args:
        network: (n x n) adjacency matrix (with negative diagonal).
        n_samples: Number of perturbation experiments.
        snr: Signal-to-noise ratio (GeneSpider convention). Higher = cleaner.
        seed: Random seed.

    Returns:
        (n_samples x n_genes) expression matrix.
    """
    from scipy.stats import chi2 as chi2_dist

    rng = np.random.default_rng(seed)
    n = network.shape[0]

    # G = (I - A)^{-1}
    G = np.linalg.inv(np.eye(n) - network)

    # Perturbation matrix: sparse, one perturbation per experiment
    # GeneSpider uses identity-like P (each experiment perturbs one gene)
    P = np.zeros((n, n_samples))
    for j in range(n_samples):
        gene = j % n  # cycle through genes
        P[gene, j] = 1.0

    signal = G @ P

    # GeneSpider SNR formula
    alpha = 0.05
    s1 = np.linalg.svd(signal, compute_uv=False)[0]
    chi2_val = float(chi2_dist.ppf(1 - alpha, P.size))
    noise_std = float(s1 / (snr * np.sqrt(chi2_val))) if snr > 0 else 0.0

    noise = (
        rng.normal(0, noise_std, signal.shape)
        if noise_std > 0
        else np.zeros_like(signal)
    )

    Y = signal + noise
    return Y.T  # (n_samples x n_genes)
