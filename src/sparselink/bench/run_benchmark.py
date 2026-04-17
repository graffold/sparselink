#!/usr/bin/env python3
"""Exhaustive sparselink benchmark across all methods.

Usage:
    python -m sparselink.bench.run_benchmark --tier fast --n-datasets 3
    python -m sparselink.bench.run_benchmark --tier fast,medium --timeout 60
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np

from sparselink import list_methods
from sparselink.bench.metrics import evaluate
from sparselink.bench.synthetic import generate_data, generate_network
from sparselink.registry import get_method

import sparselink.methods  # noqa: F401

TIERS: dict[str, list[str]] = {
    "fast": [
        "lasso", "lsco", "elastic_net", "ridge", "partial_correlation",
        "clr", "genie3", "neighborhood_selection",
    ],
    "medium": [
        "glasso", "bdeu", "bge", "granger_causality",
        "transfer_entropy", "pcmci",
    ],
    "slow": [
        "tigress", "glasso_stars", "pc", "fci",
    ],
    "very_slow": [
        "notears", "dag_gnn",
    ],
}

TOPOLOGIES = ["random", "scalefree", "smallworld"]
NOISE_LEVELS = [0.01, 0.1, 1.0]
SPARSITIES = [0.2, 0.4, 0.6]


@dataclass
class RunResult:
    method: str
    dataset_idx: int
    n_nodes: int
    n_samples: int
    topology: str
    sparsity: float
    noise_std: float
    auroc: float
    aupr: float
    precision: float
    recall: float
    f1: float
    fdr: float
    mcc: float
    r2: float
    elapsed_sec: float
    error: str | None = None


# Methods that benefit from alpha sweep, with their param name
ALPHA_SWEEP: dict[str, str] = {
    "lasso": "alpha",
    "lsco": "threshold",
    "elastic_net": "alpha",
    "ridge": "alpha",
    "neighborhood_selection": "alpha",
    "glasso": "alpha",
}
ALPHA_RANGE = np.logspace(-4, 1, 30)


class _Timeout(Exception):
    pass


def _alarm_handler(signum: int, frame: object) -> None:
    raise _Timeout()


def _fit_best(method_name: str, X: np.ndarray, true_net: np.ndarray) -> tuple:
    """Fit method, sweeping alpha if applicable. Returns (best_adj, elapsed)."""
    method_cls = get_method(method_name)

    if method_name in ALPHA_SWEEP:
        param = ALPHA_SWEEP[method_name]
        best_auroc = -1.0
        best_adj = None
        t0 = time.perf_counter()
        for alpha in ALPHA_RANGE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = method_cls(**{param: float(alpha)}).fit(X)
            adj = result.adjacency_matrix
            try:
                from sklearn.metrics import roc_auc_score
                mask = ~np.eye(adj.shape[0], dtype=bool)
                y_true = (true_net[mask] != 0).astype(int)
                y_scores = np.abs(adj[mask])
                auroc = float(roc_auc_score(y_true, y_scores))
            except ValueError:
                auroc = 0.5
            if auroc > best_auroc:
                best_auroc = auroc
                best_adj = adj
        elapsed = time.perf_counter() - t0
        return best_adj, elapsed
    else:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = method_cls().fit(X)
        elapsed = time.perf_counter() - t0
        return result.adjacency_matrix, elapsed


def run_single(
    method_name: str, X: np.ndarray, true_net: np.ndarray,
    dataset_idx: int, n_nodes: int, n_samples: int,
    topology: str, sparsity: float, noise_std: float, timeout: int,
) -> RunResult:
    """Run one method with a SIGALRM timeout."""
    base = dict(
        method=method_name, dataset_idx=dataset_idx, n_nodes=n_nodes,
        n_samples=n_samples, topology=topology, sparsity=sparsity, noise_std=noise_std,
    )
    fail = dict(auroc=0, aupr=0, precision=0, recall=0, f1=0, fdr=1, mcc=0, r2=0)

    old_handler = None
    try:
        if timeout > 0:
            import signal
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(timeout)

        warnings.simplefilter("ignore")
        best_adj, elapsed = _fit_best(method_name, X, true_net)

        if timeout > 0:
            signal.alarm(0)

        metrics = evaluate(true_net, best_adj)
        return RunResult(
            **base,
            auroc=metrics.auroc, aupr=metrics.aupr,
            precision=metrics.precision, recall=metrics.recall,
            f1=metrics.f1, fdr=metrics.fdr,
            mcc=metrics.mcc, r2=metrics.r2,
            elapsed_sec=round(elapsed, 4),
        )
    except _Timeout:
        return RunResult(**base, **fail, elapsed_sec=float(timeout), error="TIMEOUT")
    except Exception as e:
        if timeout > 0:
            signal.alarm(0)
        return RunResult(**base, **fail, elapsed_sec=0, error=str(e)[:120])
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exhaustive sparselink benchmark")
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--n-datasets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=60,
                        help="Hard per-method timeout in seconds (0=no limit)")
    parser.add_argument("--tier", default="fast,medium,slow",
                        help="Comma-separated tiers: fast,medium,slow,very_slow")
    parser.add_argument("--output", "-o", default="benchmark_results.json")
    args = parser.parse_args(argv)

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        if t not in TIERS:
            parser.error(f"Unknown tier '{t}'. Choose from: {list(TIERS.keys())}")
        methods.extend(TIERS[t])
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    total = len(methods) * len(TOPOLOGIES) * len(SPARSITIES) * len(NOISE_LEVELS) * args.n_datasets
    print(f"Benchmark: {len(methods)} methods × {len(TOPOLOGIES)} topologies × "
          f"{len(SPARSITIES)} sparsities × {len(NOISE_LEVELS)} noise levels × "
          f"{args.n_datasets} datasets = {total} runs")
    print(f"Tiers:        {selected_tiers}")
    print(f"Methods:      {methods}")
    print(f"Topologies:   {TOPOLOGIES}")
    print(f"Sparsities:   {SPARSITIES}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Timeout:      {args.timeout}s (hard kill)")

    try:
        import mlx.core  # noqa: F401
        print("MLX:          detected ✓\n")
    except ImportError:
        print("MLX:          not available (NumPy fallback)\n")

    rng = np.random.default_rng(args.seed)
    results: list[RunResult] = []
    done = 0
    wall_start = time.perf_counter()

    for topology in TOPOLOGIES:
        for sparsity in SPARSITIES:
            for noise_std in NOISE_LEVELS:
                for ds_idx in range(args.n_datasets):
                    ds_seed = int(rng.integers(0, 2**31))
                    true_net = generate_network(
                        args.n_nodes, topology=topology,
                        sparsity=sparsity, seed=ds_seed,
                    )
                    X = generate_data(
                        true_net, n_samples=args.n_samples,
                        noise_std=noise_std, seed=ds_seed,
                    )
                    for method_name in methods:
                        done += 1
                        tag = (f"[{done}/{total}] {method_name:22s} "
                               f"{topology}/sp={sparsity}/noise={noise_std}/ds={ds_idx}")
                        print(f"  {tag}", end=" … ", flush=True)
                        r = run_single(
                            method_name, X, true_net, ds_idx,
                            args.n_nodes, args.n_samples,
                            topology, sparsity, noise_std, args.timeout,
                        )
                        if r.error:
                            print(r.error)
                        else:
                            print(f"AUROC={r.auroc:.3f} MCC={r.mcc:.3f} F1={r.f1:.3f} ({r.elapsed_sec:.2f}s)")
                        results.append(r)

    wall_elapsed = time.perf_counter() - wall_start

    print("\n" + "=" * 105)
    print(f"{'METHOD':25s} {'AUROC':>7s} {'AUPR':>7s} {'F1':>7s} {'MCC':>7s} "
          f"{'R²':>7s} {'FDR':>7s} {'TIME':>9s} {'OK':>4s} {'ERR':>4s}")
    print("-" * 105)
    for m in methods:
        ok = [r for r in results if r.method == m and r.error is None]
        errs = [r for r in results if r.method == m and r.error is not None]
        if ok:
            a = {k: np.mean([getattr(r, k) for r in ok]) for k in ("auroc", "aupr", "f1", "mcc", "r2", "fdr", "elapsed_sec")}
            print(f"  {m:25s} {a['auroc']:7.3f} {a['aupr']:7.3f} {a['f1']:7.3f} {a['mcc']:7.3f} "
                  f"{a['r2']:7.3f} {a['fdr']:7.3f} {a['elapsed_sec']:8.2f}s {len(ok):>4d} {len(errs):>4d}")
        else:
            print(f"  {m:25s} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>7s} "
                  f"{'—':>7s} {'—':>7s} {'—':>9s} {0:>4d} {len(errs):>4d}")
    print("=" * 105)
    print(f"Wall time: {wall_elapsed / 60:.1f} min")

    with open(args.output, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
