"""Tests for sparselink.bench: synthetic data, metrics, and runner."""

from __future__ import annotations

import numpy as np

from sparselink.bench.metrics import MetricsResult, evaluate
from sparselink.bench.runner import BenchmarkConfig, run_benchmark
from sparselink.bench.synthetic import generate_data, generate_network


# --- synthetic ---


class TestGenerateNetwork:
    def test_random_shape(self) -> None:
        A = generate_network(10, topology="random", sparsity=0.3, seed=0)
        assert A.shape == (10, 10)

    def test_zero_diagonal(self) -> None:
        A = generate_network(10, seed=2)
        np.testing.assert_array_equal(np.diag(A), 0)

    def test_scalefree_shape(self) -> None:
        A = generate_network(15, topology="scalefree", sparsity=0.3, seed=3)
        assert A.shape == (15, 15)

    def test_sparsity_affects_density(self) -> None:
        sparse = generate_network(20, sparsity=0.1, seed=5)
        dense = generate_network(20, sparsity=0.5, seed=5)
        assert np.count_nonzero(sparse) < np.count_nonzero(dense)


class TestGenerateData:
    def test_shape(self) -> None:
        A = generate_network(10, seed=0)
        X = generate_data(A, n_samples=50, seed=0)
        assert X.shape == (50, 10)

    def test_deterministic(self) -> None:
        A = generate_network(5, seed=0)
        X1 = generate_data(A, n_samples=30, seed=1)
        X2 = generate_data(A, n_samples=30, seed=1)
        np.testing.assert_array_equal(X1, X2)

    def test_zero_network(self) -> None:
        A = np.zeros((5, 5))
        X = generate_data(A, n_samples=20, noise_std=0.1, seed=0)
        assert X.shape == (20, 5)


# --- metrics ---


class TestEvaluate:
    def test_perfect_prediction(self) -> None:
        true = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        result = evaluate(true, true)
        assert isinstance(result, MetricsResult)
        assert result.auroc == 1.0
        assert result.aupr == 1.0

    def test_zero_prediction(self) -> None:
        true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        pred = np.zeros((3, 3))
        result = evaluate(true, pred, threshold=0.5)
        assert result.recall == 0.0

    def test_random_prediction_auroc_around_half(self) -> None:
        rng = np.random.default_rng(42)
        true = (rng.random((20, 20)) > 0.8).astype(float)
        np.fill_diagonal(true, 0)
        pred = rng.random((20, 20))
        np.fill_diagonal(pred, 0)
        result = evaluate(true, pred)
        assert 0.3 < result.auroc < 0.7

    def test_explicit_threshold(self) -> None:
        true = np.array([[0, 1], [1, 0]], dtype=float)
        pred = np.array([[0, 0.8], [0.8, 0]], dtype=float)
        result = evaluate(true, pred, threshold=0.5)
        assert result.precision == 1.0
        assert result.recall == 1.0


# --- runner ---


class TestRunBenchmark:
    def test_basic_run(self) -> None:
        config = BenchmarkConfig(
            methods=["lasso", "ridge"],
            n_datasets=2,
            n_nodes=8,
            n_samples=40,
            seed=0,
        )
        results = run_benchmark(config)
        assert len(results) == 4  # 2 methods x 2 datasets
        for r in results:
            assert r.method in ("lasso", "ridge")
            assert isinstance(r.metrics, MetricsResult)

    def test_scalefree_topology(self) -> None:
        config = BenchmarkConfig(
            methods=["lasso"],
            n_datasets=1,
            n_nodes=10,
            n_samples=50,
            topology="scalefree",
            seed=1,
        )
        results = run_benchmark(config)
        assert len(results) == 1
        assert 0.0 <= results[0].metrics.auroc <= 1.0
