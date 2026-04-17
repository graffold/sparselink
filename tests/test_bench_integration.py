"""Integration tests for sparselink.bench pipeline runner."""

from __future__ import annotations

import numpy as np
import pytest

from sparselink.bench.runner import BenchmarkConfig, run_benchmark
from sparselink.bench.synthetic import generate_data, generate_network


@pytest.mark.integration
class TestPipelineRunnerIntegration:
    """End-to-end integration tests for the benchmark pipeline."""

    def test_multiple_methods_multiple_datasets(self) -> None:
        """Run 3 methods on 3 datasets and verify all results collected."""
        config = BenchmarkConfig(
            methods=["lasso", "ridge", "elastic_net"],
            n_datasets=3,
            n_nodes=10,
            n_samples=50,
            seed=42,
        )
        results = run_benchmark(config)
        assert len(results) == 9  # 3 methods x 3 datasets
        methods_seen = {r.method for r in results}
        assert methods_seen == {"lasso", "ridge", "elastic_net"}
        datasets_seen = {r.dataset_idx for r in results}
        assert datasets_seen == {0, 1, 2}

    def test_results_have_valid_metrics(self) -> None:
        """All metrics should be in valid ranges."""
        config = BenchmarkConfig(
            methods=["lasso", "partial_correlation"],
            n_datasets=2,
            n_nodes=10,
            n_samples=60,
            seed=7,
        )
        results = run_benchmark(config)
        for r in results:
            assert 0.0 <= r.metrics.auroc <= 1.0
            assert 0.0 <= r.metrics.aupr <= 1.0
            assert 0.0 <= r.metrics.precision <= 1.0
            assert 0.0 <= r.metrics.recall <= 1.0
            assert 0.0 <= r.metrics.fdr <= 1.0

    def test_reproducibility(self) -> None:
        """Same seed produces same results."""
        config = BenchmarkConfig(
            methods=["lasso"],
            n_datasets=2,
            n_nodes=8,
            n_samples=40,
            seed=123,
        )
        r1 = run_benchmark(config)
        r2 = run_benchmark(config)
        for a, b in zip(r1, r2):
            assert a.metrics.auroc == b.metrics.auroc

    def test_synthetic_data_pipeline(self) -> None:
        """Test generate_network -> generate_data -> method.fit flow."""
        net = generate_network(15, topology="scalefree", sparsity=0.2, seed=0)
        X = generate_data(net, n_samples=80, noise_std=1.0, seed=0)

        from sparselink import get_method

        method = get_method("lasso")(alpha=0.01)
        result = method.fit(X)
        assert result.adjacency_matrix.shape == (15, 15)
        assert np.count_nonzero(result.adjacency_matrix) > 0
