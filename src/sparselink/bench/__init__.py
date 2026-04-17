"""sparselink.bench - Benchmarking for network inference algorithms."""

from sparselink.bench.metrics import MetricsResult, evaluate
from sparselink.bench.runner import run_benchmark
from sparselink.bench.synthetic import generate_data, generate_network

__all__ = [
    "MetricsResult",
    "generate_network",
    "generate_data",
    "evaluate",
    "run_benchmark",
]
