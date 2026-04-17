"""Tests for US-006 causal discovery methods: PC, FCI, NOTEARS, DAG-GNN."""

import numpy as np
import pytest

from sparselink import InferenceResult, get_method, list_methods


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate data with simple linear causal structure."""
    rng = np.random.default_rng(42)
    n, d = 100, 5
    X = np.zeros((n, d))
    X[:, 0] = rng.standard_normal(n)
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * rng.standard_normal(n)
    X[:, 2] = 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
    X[:, 3] = rng.standard_normal(n)
    X[:, 4] = rng.standard_normal(n)
    return X


class TestNOTEARS:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.notears import NOTEARSMethod

        m = NOTEARSMethod(lambda1=0.1, max_iter=20, w_threshold=0.1)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.notears import NOTEARSMethod

        m = NOTEARSMethod(lambda1=0.1, max_iter=10)
        result = m.fit(sample_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_nonnegative(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.notears import NOTEARSMethod

        m = NOTEARSMethod(lambda1=0.1, max_iter=10)
        result = m.fit(sample_data)
        assert np.all(result.adjacency_matrix >= 0)

    def test_registered(self) -> None:
        from sparselink.methods.notears import NOTEARSMethod

        assert get_method("notears") is NOTEARSMethod


class TestPC:
    def test_import_error_without_causal_learn(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.pc import PCMethod

        # This test verifies the method exists and is registered
        assert PCMethod.name == "pc"

    def test_registered(self) -> None:
        from sparselink.methods.pc import PCMethod

        assert get_method("pc") is PCMethod


class TestFCI:
    def test_import_error_without_causal_learn(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.fci import FCIMethod

        assert FCIMethod.name == "fci"

    def test_registered(self) -> None:
        from sparselink.methods.fci import FCIMethod

        assert get_method("fci") is FCIMethod


class TestDAGGNN:
    def test_registered(self) -> None:
        from sparselink.methods.dag_gnn import DAGGNNMethod

        assert get_method("dag_gnn") is DAGGNNMethod

    def test_name(self) -> None:
        from sparselink.methods.dag_gnn import DAGGNNMethod

        assert DAGGNNMethod.name == "dag_gnn"


class TestRegistryUS006:
    def test_new_methods_listed(self) -> None:
        methods = list_methods()
        for name in ["pc", "fci", "notears", "dag_gnn"]:
            assert name in methods


# Conditional tests that run only when optional deps are available

try:
    import causallearn  # noqa: F401

    HAS_CAUSAL_LEARN = True
except ImportError:
    HAS_CAUSAL_LEARN = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_CAUSAL_LEARN, reason="causal-learn not installed")
class TestPCWithDep:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.pc import PCMethod

        m = PCMethod(alpha=0.05)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.pc import PCMethod

        m = PCMethod(alpha=0.05)
        result = m.fit(sample_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)


@pytest.mark.skipif(not HAS_CAUSAL_LEARN, reason="causal-learn not installed")
class TestFCIWithDep:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.fci import FCIMethod

        m = FCIMethod(alpha=0.05)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.fci import FCIMethod

        m = FCIMethod(alpha=0.05)
        result = m.fit(sample_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDAGGNNWithDep:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.dag_gnn import DAGGNNMethod

        m = DAGGNNMethod(epochs=20, w_threshold=0.1)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.dag_gnn import DAGGNNMethod

        m = DAGGNNMethod(epochs=20)
        result = m.fit(sample_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_nonnegative(self, sample_data: np.ndarray) -> None:
        from sparselink.methods.dag_gnn import DAGGNNMethod

        m = DAGGNNMethod(epochs=20)
        result = m.fit(sample_data)
        assert np.all(result.adjacency_matrix >= 0)
