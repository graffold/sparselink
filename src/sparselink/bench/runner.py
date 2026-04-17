"""Pipeline runner: run N methods on M datasets and collect metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparselink.bench.metrics import MetricsResult, evaluate
from sparselink.bench.synthetic import generate_data, generate_network
from sparselink.registry import get_method


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    methods: list[str]
    n_datasets: int = 5
    n_nodes: int = 20
    n_samples: int = 100
    topology: str = "random"
    sparsity: float = 0.2
    noise_std: float = 0.1
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Single result row."""

    method: str
    dataset_idx: int
    metrics: MetricsResult


def run_benchmark(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Run benchmark: N methods x M datasets.

    Args:
        config: BenchmarkConfig specifying methods, datasets, and parameters.

    Returns:
        List of BenchmarkResult for each (method, dataset) pair.
    """
    results: list[BenchmarkResult] = []
    rng = np.random.default_rng(config.seed)

    for ds_idx in range(config.n_datasets):
        ds_seed = int(rng.integers(0, 2**31))
        true_net = generate_network(
            config.n_nodes,
            topology=config.topology,
            sparsity=config.sparsity,
            seed=ds_seed,
        )
        X = generate_data(
            true_net,
            n_samples=config.n_samples,
            noise_std=config.noise_std,
            seed=ds_seed,
        )

        for method_name in config.methods:
            method_cls = get_method(method_name)
            method = method_cls()
            result = method.fit(X)
            metrics = evaluate(true_net, result.adjacency_matrix)
            results.append(
                BenchmarkResult(method=method_name, dataset_idx=ds_idx, metrics=metrics)
            )

    return results
