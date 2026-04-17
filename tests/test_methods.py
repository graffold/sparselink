"""Tests for sparse regression methods (US-002)."""

import numpy as np
import pytest

from sparselink import InferenceResult, get_method, list_methods
from sparselink.methods import (
    CLRMethod,
    ElasticNetMethod,
    LSCOMethod,
    LassoMethod,
    RidgeMethod,
)


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 10))


@pytest.fixture
def perturbation_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 10))
    y = rng.standard_normal((50, 10))
    return X, y


class TestLasso:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        m = LassoMethod(alpha=0.1)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_registered(self) -> None:
        cls = get_method("lasso")
        assert cls is LassoMethod


class TestLSCO:
    def test_fit_no_threshold(self, sample_data: np.ndarray) -> None:
        m = LSCOMethod(threshold=0.0)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_fit_with_threshold(self, sample_data: np.ndarray) -> None:
        m = LSCOMethod(threshold=0.5)
        result = m.fit(sample_data)
        # Some entries should be zeroed out
        assert np.any(result.adjacency_matrix == 0)

    def test_fit_with_perturbation(self, perturbation_data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = perturbation_data
        m = LSCOMethod(threshold=0.0)
        result = m.fit(X, y=y)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_registered(self) -> None:
        assert get_method("lsco") is LSCOMethod


class TestCLR:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        m = CLRMethod()
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_symmetric(self, sample_data: np.ndarray) -> None:
        m = CLRMethod()
        result = m.fit(sample_data)
        np.testing.assert_allclose(
            result.adjacency_matrix, result.adjacency_matrix.T, atol=1e-10
        )

    def test_nonnegative(self, sample_data: np.ndarray) -> None:
        m = CLRMethod()
        result = m.fit(sample_data)
        assert np.all(result.adjacency_matrix >= 0)

    def test_registered(self) -> None:
        assert get_method("clr") is CLRMethod


class TestElasticNet:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        m = ElasticNetMethod(alpha=0.1, l1_ratio=0.5)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_registered(self) -> None:
        assert get_method("elastic_net") is ElasticNetMethod


class TestRidge:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        m = RidgeMethod(alpha=1.0)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_dense_output(self, sample_data: np.ndarray) -> None:
        # Ridge should produce dense (non-sparse) output
        m = RidgeMethod(alpha=0.01)
        result = m.fit(sample_data)
        nonzero_frac = np.count_nonzero(result.adjacency_matrix) / result.adjacency_matrix.size
        assert nonzero_frac > 0.5

    def test_registered(self) -> None:
        assert get_method("ridge") is RidgeMethod


class TestRegistry:
    def test_all_methods_listed(self) -> None:
        methods = list_methods()
        for name in ["lasso", "lsco", "clr", "elastic_net", "ridge", "partial_correlation"]:
            assert name in methods
