"""sparselink.bench - Benchmarking and synthetic data for network inference."""

from sparselink.bench.metrics import MetricsResult, evaluate
from sparselink.bench.nestboot import NestBoot
from sparselink.bench.runner import run_benchmark
from sparselink.bench.synthetic import generate_expression, generate_network

__all__ = [
    "MetricsResult",
    "generate_network",
    "generate_expression",
    "evaluate",
    "NestBoot",
    "run_benchmark",
]
