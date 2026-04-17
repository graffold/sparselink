"""Tests for Bayesian structure learning methods (BDeu and BGe)."""

import numpy as np
import pytest

from sparselink.methods.bayesian import BDeuMethod, BGeMethod
from sparselink.registry import get_method, list_methods


@pytest.fixture
def causal_data():
    """Generate data with known causal structure: x0 -> x1 -> x2."""
    rng = np.random.default_rng(42)
    n = 200
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.2 * rng.standard_normal(n)
    x2 = 0.8 * x1 + 0.2 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2])


class TestBDeuMethod:
    def test_registered(self):
        assert "bdeu" in list_methods()
        assert get_method("bdeu") is BDeuMethod

    def test_fit_returns_result(self, causal_data):
        method = BDeuMethod(ess=10.0, n_bins=3, max_parents=2)
        result = method.fit(causal_data)
        assert result.adjacency_matrix.shape == (3, 3)

    def test_output_is_dag(self, causal_data):
        method = BDeuMethod(ess=10.0, n_bins=3, max_parents=2)
        result = method.fit(causal_data)
        adj = result.adjacency_matrix
        # DAG: no self-loops
        assert np.all(np.diag(adj) == 0)
        # DAG: no cycles (check via matrix power)
        n = adj.shape[0]
        power = np.eye(n)
        for _ in range(n):
            power = power @ (adj != 0).astype(float)
        assert np.all(np.diag(power) == 0)

    def test_detects_edges(self, causal_data):
        method = BDeuMethod(ess=10.0, n_bins=3, max_parents=2)
        result = method.fit(causal_data)
        # Should find at least one edge in this strongly coupled data
        assert np.any(result.adjacency_matrix != 0)

    def test_edge_list_populated(self, causal_data):
        method = BDeuMethod(ess=10.0, n_bins=3, max_parents=2)
        result = method.fit(causal_data)
        if np.any(result.adjacency_matrix != 0):
            assert len(result.edge_list) > 0

    def test_independent_data_sparse(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((100, 4))
        method = BDeuMethod(ess=10.0, n_bins=3, max_parents=2)
        result = method.fit(X)
        # Independent data should yield few or no edges
        assert result.adjacency_matrix.sum() <= 4


class TestBGeMethod:
    def test_registered(self):
        assert "bge" in list_methods()
        assert get_method("bge") is BGeMethod

    def test_fit_returns_result(self, causal_data):
        method = BGeMethod(max_parents=2)
        result = method.fit(causal_data)
        assert result.adjacency_matrix.shape == (3, 3)

    def test_output_is_dag(self, causal_data):
        method = BGeMethod(max_parents=2)
        result = method.fit(causal_data)
        adj = result.adjacency_matrix
        # No self-loops
        assert np.all(np.diag(adj) == 0)
        # No cycles
        n = adj.shape[0]
        power = np.eye(n)
        for _ in range(n):
            power = power @ (adj != 0).astype(float)
        assert np.all(np.diag(power) == 0)

    def test_detects_causal_edges(self, causal_data):
        method = BGeMethod(max_parents=2)
        result = method.fit(causal_data)
        # Should find edges in strongly coupled Gaussian data
        assert np.any(result.adjacency_matrix != 0)

    def test_edge_list_populated(self, causal_data):
        method = BGeMethod(max_parents=2)
        result = method.fit(causal_data)
        if np.any(result.adjacency_matrix != 0):
            assert len(result.edge_list) > 0

    def test_independent_data_sparse(self):
        rng = np.random.default_rng(456)
        X = rng.standard_normal((100, 4))
        method = BGeMethod(max_parents=2)
        result = method.fit(X)
        # Independent Gaussian data should yield few or no edges
        assert result.adjacency_matrix.sum() <= 4

    def test_custom_alpha_w(self, causal_data):
        method = BGeMethod(alpha_mu=1.0, alpha_w=10.0, max_parents=2)
        result = method.fit(causal_data)
        assert result.adjacency_matrix.shape == (3, 3)
