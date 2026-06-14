"""Network comparison utilities."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def jaccard_index(A: np.ndarray, B: np.ndarray, threshold: float | None = None) -> float:
    """Jaccard index of edge sets. Binarizes at threshold (default: median nonzero)."""
    def _binarize(M, t):
        nz = np.abs(M[M != 0])
        thr = t if t is not None else (float(np.median(nz)) if len(nz) > 0 else 0.0)
        return (np.abs(M) > thr).astype(int)
    a, b = _binarize(A, threshold), _binarize(B, threshold)
    np.fill_diagonal(a, 0); np.fill_diagonal(b, 0)
    intersection = np.sum(a & b)
    union = np.sum(a | b)
    return float(intersection / union) if union > 0 else 0.0


def edge_rank_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman correlation of off-diagonal edge weights."""
    mask = ~np.eye(A.shape[0], dtype=bool)
    return float(spearmanr(np.abs(A[mask]), np.abs(B[mask]))[0])


def shared_edges(A: np.ndarray, B: np.ndarray, top_k: int = 100) -> set[tuple[int, int]]:
    """Return (i,j) pairs in top_k edges of both networks."""
    mask = ~np.eye(A.shape[0], dtype=bool)
    def _top_k(M):
        flat = np.abs(M[mask])
        indices = np.argsort(-flat)[:top_k]
        rows, cols = np.where(mask)
        return {(rows[i], cols[i]) for i in indices}
    return _top_k(A) & _top_k(B)


def network_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius norm of difference between adjacency matrices."""
    return float(np.linalg.norm(A - B, 'fro'))
