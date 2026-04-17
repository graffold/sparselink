"""Tests for correlation/info-theoretic methods (US-003)."""

import numpy as np
import pytest

from sparselink import InferenceResult, get_method, list_methods
from sparselink.methods import (
    GrangerCausality,
    PCMCIMethod,
    PartialCorrelation,
    TransferEntropy,
)


@pytest.fixture
def timeseries_data() -> np.ndarray:
    """Generate simple time-series with causal structure: x0 -> x1."""
    rng = np.random.default_rng(42)
    T, n = 200, 5
    X = np.zeros((T, n))
    X[0] = rng.standard_normal(n)
    for t in range(1, T):
        X[t, 0] = 0.8 * X[t - 1, 0] + 0.2 * rng.standard_normal()
        X[t, 1] = 0.5 * X[t - 1, 0] + 0.3 * rng.standard_normal()
        X[t, 2:] = 0.3 * rng.standard_normal(n - 2)
    return X


@pytest.fixture
def sample_data() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 10))


class TestPartialCorrelation:
    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        m = PartialCorrelation(threshold=0.0)
        result = m.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (10, 10)

    def test_zero_diagonal(self, sample_data: np.ndarray) -> None:
        m = PartialCorrelation()
        result = m.fit(sample_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_threshold(self, sample_data: np.ndarray) -> None:
        m = PartialCorrelation(threshold=0.5)
        result = m.fit(sample_data)
        nonzero = result.adjacency_matrix[result.adjacency_matrix != 0]
        assert np.all(np.abs(nonzero) >= 0.5)

    def test_registered(self) -> None:
        assert get_method("partial_correlation") is PartialCorrelation


class TestPCMCI:
    def test_fit_returns_result(self, timeseries_data: np.ndarray) -> None:
        m = PCMCIMethod(max_lag=2)
        result = m.fit(timeseries_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, timeseries_data: np.ndarray) -> None:
        m = PCMCIMethod(max_lag=1)
        result = m.fit(timeseries_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_detects_causal_link(self, timeseries_data: np.ndarray) -> None:
        m = PCMCIMethod(max_lag=1)
        result = m.fit(timeseries_data)
        # x0 -> x1 should be among the strongest links
        assert result.adjacency_matrix[0, 1] > 0

    def test_registered(self) -> None:
        assert get_method("pcmci") is PCMCIMethod


class TestGrangerCausality:
    def test_fit_returns_result(self, timeseries_data: np.ndarray) -> None:
        m = GrangerCausality(max_lag=2)
        result = m.fit(timeseries_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, timeseries_data: np.ndarray) -> None:
        m = GrangerCausality(max_lag=1)
        result = m.fit(timeseries_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_detects_causal_link(self, timeseries_data: np.ndarray) -> None:
        m = GrangerCausality(max_lag=1)
        result = m.fit(timeseries_data)
        # x0 -> x1 should have high F-stat
        assert result.adjacency_matrix[0, 1] > result.adjacency_matrix[2, 1]

    def test_registered(self) -> None:
        assert get_method("granger_causality") is GrangerCausality


class TestTransferEntropy:
    def test_fit_returns_result(self, timeseries_data: np.ndarray) -> None:
        m = TransferEntropy(max_lag=1, n_bins=8)
        result = m.fit(timeseries_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_zero_diagonal(self, timeseries_data: np.ndarray) -> None:
        m = TransferEntropy(max_lag=1)
        result = m.fit(timeseries_data)
        np.testing.assert_array_equal(np.diag(result.adjacency_matrix), 0.0)

    def test_nonnegative(self, timeseries_data: np.ndarray) -> None:
        m = TransferEntropy(max_lag=1)
        result = m.fit(timeseries_data)
        assert np.all(result.adjacency_matrix >= 0)

    def test_detects_causal_link(self, timeseries_data: np.ndarray) -> None:
        m = TransferEntropy(max_lag=1, n_bins=10)
        result = m.fit(timeseries_data)
        # x0 -> x1 should have higher TE than noise variables
        assert result.adjacency_matrix[0, 1] > result.adjacency_matrix[3, 4]

    def test_registered(self) -> None:
        assert get_method("transfer_entropy") is TransferEntropy


class TestRegistryUS003:
    def test_new_methods_listed(self) -> None:
        methods = list_methods()
        for name in ["partial_correlation", "pcmci", "granger_causality", "transfer_entropy"]:
            assert name in methods
