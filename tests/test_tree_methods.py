"""Tests for tree-based methods (GENIE3 and TIGRESS)."""

import numpy as np
import pandas as pd
import pytest

import sparselink
from sparselink import get_method, list_methods
from sparselink.methods.genie3 import GENIE3Method
from sparselink.methods.tigress import TIGRESSMethod
from sparselink.types import InferenceResult


@pytest.fixture
def sample_data() -> np.ndarray:
    """Generate synthetic data with known structure."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 50, 5
    X = rng.randn(n_samples, n_features)
    # Inject dependency: feature 1 depends on feature 0
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * rng.randn(n_samples)
    return X


# --- GENIE3 Tests ---


class TestGENIE3:
    def test_registered(self) -> None:
        assert "genie3" in list_methods()
        assert get_method("genie3") is GENIE3Method

    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        method = GENIE3Method(n_estimators=10)
        result = method.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_non_negative_importances(self, sample_data: np.ndarray) -> None:
        method = GENIE3Method(n_estimators=10)
        result = method.fit(sample_data)
        assert np.all(result.adjacency_matrix >= 0)

    def test_diagonal_zero(self, sample_data: np.ndarray) -> None:
        method = GENIE3Method(n_estimators=10)
        result = method.fit(sample_data)
        assert np.allclose(np.diag(result.adjacency_matrix), 0)

    def test_detects_dependency(self, sample_data: np.ndarray) -> None:
        method = GENIE3Method(n_estimators=50, random_state=0)
        result = method.fit(sample_data)
        # Feature 0 should be important for predicting feature 1
        assert result.adjacency_matrix[0, 1] > 0.1

    def test_accepts_dataframe(self, sample_data: np.ndarray) -> None:
        df = pd.DataFrame(sample_data)
        method = GENIE3Method(n_estimators=10)
        result = method.fit(df)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_edge_list_populated(self, sample_data: np.ndarray) -> None:
        method = GENIE3Method(n_estimators=10)
        result = method.fit(sample_data)
        assert len(result.edge_list) > 0


# --- TIGRESS Tests ---


class TestTIGRESS:
    def test_registered(self) -> None:
        assert "tigress" in list_methods()
        assert get_method("tigress") is TIGRESSMethod

    def test_fit_returns_result(self, sample_data: np.ndarray) -> None:
        method = TIGRESSMethod(n_bootstrap=10)
        result = method.fit(sample_data)
        assert isinstance(result, InferenceResult)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_scores_in_zero_one(self, sample_data: np.ndarray) -> None:
        method = TIGRESSMethod(n_bootstrap=10)
        result = method.fit(sample_data)
        assert np.all(result.adjacency_matrix >= 0)
        assert np.all(result.adjacency_matrix <= 1)

    def test_diagonal_zero(self, sample_data: np.ndarray) -> None:
        method = TIGRESSMethod(n_bootstrap=10)
        result = method.fit(sample_data)
        assert np.allclose(np.diag(result.adjacency_matrix), 0)

    def test_detects_dependency(self, sample_data: np.ndarray) -> None:
        method = TIGRESSMethod(n_bootstrap=20, random_state=0)
        result = method.fit(sample_data)
        # Feature 0 should be frequently selected for predicting feature 1
        assert result.adjacency_matrix[0, 1] > 0.3

    def test_accepts_dataframe(self, sample_data: np.ndarray) -> None:
        df = pd.DataFrame(sample_data)
        method = TIGRESSMethod(n_bootstrap=10)
        result = method.fit(df)
        assert result.adjacency_matrix.shape == (5, 5)

    def test_edge_list_populated(self, sample_data: np.ndarray) -> None:
        method = TIGRESSMethod(n_bootstrap=10)
        result = method.fit(sample_data)
        assert len(result.edge_list) > 0
