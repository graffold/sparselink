"""Tests for graphical model methods (US-004)."""

import numpy as np
import pytest

from sparselink import get_method, list_methods
from sparselink.methods.glasso import GLASSOStARS, GraphicalLassoMethod
from sparselink.methods.neighborhood import NeighborhoodSelection
from sparselink.types import InferenceResult


@pytest.fixture
def correlated_data() -> np.ndarray:
    """Generate data with known sparse structure (features 0-1 correlated)."""
    rng = np.random.default_rng(0)
    n = 100
    x0 = rng.normal(size=n)
    x1 = x0 * 0.8 + rng.normal(size=n) * 0.2
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    return np.column_stack([x0, x1, x2, x3])


class TestGraphicalLasso:
    def test_registered(self) -> None:
        assert "glasso" in list_methods()
        assert get_method("glasso") is GraphicalLassoMethod

    def test_fit_returns_result(self, correlated_data: np.ndarray) -> None:
        method = GraphicalLassoMethod(alpha=0.3)
        result = method.fit(correlated_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (4, 4)

    def test_diagonal_is_zero(self, correlated_data: np.ndarray) -> None:
        result = GraphicalLassoMethod(alpha=0.3).fit(correlated_data)
        assert np.allclose(np.diag(result.adjacency_matrix), 0.0)

    def test_symmetric(self, correlated_data: np.ndarray) -> None:
        result = GraphicalLassoMethod(alpha=0.3).fit(correlated_data)
        assert np.allclose(result.adjacency_matrix, result.adjacency_matrix.T)

    def test_detects_correlation(self, correlated_data: np.ndarray) -> None:
        result = GraphicalLassoMethod(alpha=0.1).fit(correlated_data)
        # Edge between 0 and 1 should be strongest
        assert result.adjacency_matrix[0, 1] > result.adjacency_matrix[0, 2]


class TestGLASSOStARS:
    def test_registered(self) -> None:
        assert "glasso_stars" in list_methods()
        assert get_method("glasso_stars") is GLASSOStARS

    def test_fit_returns_result(self, correlated_data: np.ndarray) -> None:
        method = GLASSOStARS(n_alphas=3, n_subsamples=5)
        result = method.fit(correlated_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (4, 4)

    def test_metadata_has_alpha(self, correlated_data: np.ndarray) -> None:
        method = GLASSOStARS(n_alphas=3, n_subsamples=5)
        result = method.fit(correlated_data)
        assert "selected_alpha" in result.metadata

    def test_diagonal_is_zero(self, correlated_data: np.ndarray) -> None:
        result = GLASSOStARS(n_alphas=3, n_subsamples=5).fit(correlated_data)
        assert np.allclose(np.diag(result.adjacency_matrix), 0.0)


class TestNeighborhoodSelection:
    def test_registered(self) -> None:
        assert "neighborhood_selection" in list_methods()
        assert get_method("neighborhood_selection") is NeighborhoodSelection

    def test_fit_returns_result(self, correlated_data: np.ndarray) -> None:
        method = NeighborhoodSelection(alpha=0.05)
        result = method.fit(correlated_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (4, 4)

    def test_and_rule_symmetric(self, correlated_data: np.ndarray) -> None:
        result = NeighborhoodSelection(alpha=0.05, rule="and").fit(correlated_data)
        assert np.allclose(result.adjacency_matrix, result.adjacency_matrix.T)

    def test_or_rule_symmetric(self, correlated_data: np.ndarray) -> None:
        result = NeighborhoodSelection(alpha=0.05, rule="or").fit(correlated_data)
        assert np.allclose(result.adjacency_matrix, result.adjacency_matrix.T)

    def test_binary_output(self, correlated_data: np.ndarray) -> None:
        result = NeighborhoodSelection(alpha=0.05).fit(correlated_data)
        unique_vals = set(np.unique(result.adjacency_matrix))
        assert unique_vals <= {0.0, 1.0}

    def test_detects_edge(self, correlated_data: np.ndarray) -> None:
        result = NeighborhoodSelection(alpha=0.01).fit(correlated_data)
        # Should detect edge between correlated features 0 and 1
        assert result.adjacency_matrix[0, 1] > 0
