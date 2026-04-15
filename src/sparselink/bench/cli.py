"""CLI entry point for batch benchmarking."""

from __future__ import annotations

import argparse
import json

from sparselink.bench.runner import BenchmarkConfig, run_benchmark
from sparselink.registry import list_methods


def main(argv: list[str] | None = None) -> None:
    """CLI for sparselink-bench."""
    parser = argparse.ArgumentParser(
        prog="sparselink-bench",
        description="Run batch benchmarking of network inference methods.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method names to benchmark (default: all registered).",
    )
    parser.add_argument(
        "--n-datasets", type=int, default=5, help="Number of synthetic datasets."
    )
    parser.add_argument("--n-genes", type=int, default=20, help="Genes per dataset.")
    parser.add_argument(
        "--n-samples", type=int, default=100, help="Samples per dataset."
    )
    parser.add_argument("--topology", default="random", choices=["random", "scalefree"])
    parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", "-o", default=None, help="Output JSON file (default: stdout)."
    )

    args = parser.parse_args(argv)

    import sparselink.methods  # noqa: F401 — ensure methods registered

    methods = args.methods or list_methods()

    config = BenchmarkConfig(
        methods=methods,
        n_datasets=args.n_datasets,
        n_genes=args.n_genes,
        n_samples=args.n_samples,
        topology=args.topology,
        sparsity=args.sparsity,
        snr=args.snr,
        seed=args.seed,
    )

    results = run_benchmark(config)

    rows = [
        {
            "method": r.method,
            "dataset": r.dataset_idx,
            "auroc": r.metrics.auroc,
            "aupr": r.metrics.aupr,
            "precision": r.metrics.precision,
            "recall": r.metrics.recall,
            "f1": r.metrics.f1,
            "fdr": r.metrics.fdr,
            "mcc": r.metrics.mcc,
            "r2": r.metrics.r2,
        }
        for r in results
    ]

    output = json.dumps(rows, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
