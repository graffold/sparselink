#!/usr/bin/env python3
"""sparselink — CLI for sparse network inference.

Usage::

    sparselink                                  # interactive mode
    sparselink status                           # check methods, MLX, deps
    sparselink infer data.csv --method lasso
    sparselink bench --tier fast
    sparselink show results.json
    sparselink methods                          # list all methods
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

console = Console()

# ── Network Graph palette ─────────────────────────────────────────────────
TEAL = "bold #14B8A6"
INDIGO = "bold #818CF8"
GREEN = "#22C55E"
ORANGE = "#FB923C"
ROSE = "#FB7185"
TEXT = "#FAFAFA"
DIM = "#71717A"
BOLD = f"bold {TEXT}"

BANNER_SPARSE = [
    "███████ ██████   █████  ██████  ███████ ███████",
    "██      ██   ██ ██   ██ ██   ██ ██      ██     ",
    "███████ ██████  ███████ ██████  ███████ █████  ",
    "     ██ ██      ██   ██ ██   ██      ██ ██     ",
    "███████ ██      ██   ██ ██   ██ ███████ ███████",
]
BANNER_LINK = [
    "██      ██ ███    ██ ██   ██",
    "██      ██ ████   ██ ██  ██ ",
    "██      ██ ██ ██  ██ █████  ",
    "██      ██ ██  ██ ██ ██  ██ ",
    "███████ ██ ██   ████ ██   ██",
]


def _print_banner() -> None:
    for a, b in zip(BANNER_SPARSE, BANNER_LINK):
        console.print(f"  [{TEAL}]{a}[/] [{INDIGO}]{b}[/]")
    console.print(f"  [{BOLD}]sparselink[/]  [{DIM}]network inference toolkit[/]")
    console.print()


# ── Helpers ───────────────────────────────────────────────────────────────

def _color(val: float, lo: float = 0.4, hi: float = 0.7) -> str:
    if val >= hi:
        return GREEN
    if val >= lo:
        return ORANGE
    return ROSE


def _bar(value: float, width: int = 20) -> str:
    v = max(0.0, min(1.0, value))
    filled = int(v * width)
    return f"[{GREEN}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/]"


def _pick_multi(label: str, options: dict[str, str], default: str = "") -> list[str]:
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label} (comma-separated, or enter values directly)[/]", default=default)
    tokens = [t.strip() for t in raw.split(",")]
    result: list[str] = []
    for t in tokens:
        if t in options:
            result.append(options[t])
        elif t:
            result.append(t)
    return result


def _pick_one(label: str, options: dict[str, str], default: str = "") -> str:
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label} (or enter a value directly)[/]", default=default)
    raw = raw.strip()
    return options.get(raw, raw if raw else options.get(default, ""))


# ── Render results ────────────────────────────────────────────────────────

def _render_results(data: list[dict], title: str = "Results") -> None:
    if not data:
        console.print("[dim]No results[/]")
        return

    ok = [r for r in data if not r.get("error")]
    errs = [r for r in data if r.get("error")]
    methods = sorted(set(r["method"] for r in ok))
    metrics = ["auroc", "aupr", "f1", "mcc"]

    t = Table(title=title, title_style=TEAL, border_style="dim")
    t.add_column("Method", style="bold")
    for m in metrics:
        t.add_column(m.upper(), justify="right")
    t.add_column("", min_width=22)
    t.add_column("Time", justify="right")
    t.add_column("Runs", justify="right")

    for method in methods:
        rows = [r for r in ok if r["method"] == method]
        avgs = {m: np.mean([r.get(m, 0) for r in rows]) for m in metrics}
        avg_time = np.mean([r.get("elapsed_sec", 0) for r in rows])
        t.add_row(
            method,
            *[f"[{_color(avgs[m])}]{avgs[m]:.3f}[/]" for m in metrics],
            _bar(avgs["auroc"]),
            f"{avg_time:.2f}s",
            str(len(rows)),
        )
    console.print(t)
    if errs:
        console.print(f"  [{DIM}]{len(errs)} errors skipped[/]")

    # SNR breakdown
    snrs = sorted(set(r.get("snr", 0) for r in ok))
    if len(snrs) > 1:
        console.print()
        for snr in snrs:
            sub = [r for r in ok if r.get("snr") == snr]
            st = Table(title=f"SNR={snr}", title_style=f"bold {ORANGE}", border_style="dim", show_edge=False)
            st.add_column("Method", style="bold", width=25)
            st.add_column("AUROC", justify="right")
            st.add_column("F1", justify="right")
            st.add_column("MCC", justify="right")
            st.add_column("", min_width=22)
            for method in methods:
                rows = [r for r in sub if r["method"] == method]
                if not rows:
                    continue
                a = np.mean([r.get("auroc", 0) for r in rows])
                f = np.mean([r.get("f1", 0) for r in rows])
                m = np.mean([r.get("mcc", 0) for r in rows])
                st.add_row(method, f"[{_color(a)}]{a:.3f}[/]", f"[{_color(f)}]{f:.3f}[/]",
                           f"[{_color(m, 0.2, 0.5)}]{m:.3f}[/]", _bar(a))
            console.print(st)


# ── Infer command ─────────────────────────────────────────────────────────

def _cmd_infer(args: argparse.Namespace) -> None:
    """Infer a network from a data file."""
    import warnings
    warnings.simplefilter("ignore")
    from sparselink import get_method, list_methods
    import sparselink.methods  # noqa: F401

    # Load data
    path = Path(args.file)
    if path.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)
    elif path.suffix in (".tsv", ".txt"):
        import pandas as pd
        df = pd.read_csv(path, sep="\t")
        X = df.select_dtypes(include=[np.number]).values
        feature_names = list(df.select_dtypes(include=[np.number]).columns)
    elif path.suffix == ".npy":
        X = np.load(path)
        feature_names = [f"V{i}" for i in range(X.shape[1])]
    else:
        console.print(f"[{ROSE}]Unsupported format: {path.suffix}. Use .csv, .tsv, .txt, or .npy[/]")
        return

    console.print(f"  [{TEAL}]Data[/]    {path.name}: {X.shape[0]} samples × {X.shape[1]} features")
    console.print(f"  [{TEAL}]Method[/]  {args.method}")

    method = get_method(args.method)
    with console.status(f"[{TEAL}]Running {args.method}...[/]"):
        result = method().fit(X)

    adj = result.adjacency_matrix
    n_edges = np.count_nonzero(adj) - np.count_nonzero(np.diag(adj))
    console.print(f"  [{GREEN}]✓[/] {n_edges} edges inferred")

    # Show top edges
    edges = result.edge_list if hasattr(result, "edge_list") and result.edge_list else []
    if not edges:
        mask = ~np.eye(adj.shape[0], dtype=bool)
        idx = np.argsort(-np.abs(adj[mask]))
        rows_i, cols_j = np.where(mask)
        edges = [(rows_i[i], cols_j[i], adj[rows_i[i], cols_j[i]]) for i in idx[:20]]

    t = Table(title="Top Edges", title_style=TEAL, border_style="dim")
    t.add_column("Source", style="bold")
    t.add_column("Target", style="bold")
    t.add_column("Weight", justify="right")
    for src, tgt, w in edges[:15]:
        sn = feature_names[src] if src < len(feature_names) else str(src)
        tn = feature_names[tgt] if tgt < len(feature_names) else str(tgt)
        t.add_row(sn, tn, f"{w:.4f}")
    console.print(t)

    # Save
    if args.output:
        np.save(args.output, adj)
        console.print(f"  [{DIM}]Adjacency saved to {args.output}[/]")


# ── Methods command ───────────────────────────────────────────────────────

def _cmd_methods(args: argparse.Namespace) -> None:
    from sparselink import list_methods
    import sparselink.methods  # noqa: F401

    t = Table(title="Available Methods", title_style=TEAL, border_style="dim")
    t.add_column("Name", style="bold")
    t.add_column("Category", style=DIM)

    categories = {
        "lasso": "regression", "elastic_net": "regression", "ridge": "regression",
        "lsco": "regression", "tigress": "stability selection",
        "genie3": "tree-based", "clr": "information theory",
        "partial_correlation": "correlation",
        "glasso": "graphical model", "glasso_stars": "graphical model",
        "neighborhood_selection": "graphical model",
        "pcmci": "causal (time-series)", "granger_causality": "causal (time-series)",
        "transfer_entropy": "causal (time-series)",
        "pc": "constraint-based", "fci": "constraint-based",
        "notears": "continuous optimization", "dag_gnn": "deep learning",
        "bdeu": "bayesian", "bge": "bayesian",
    }
    for m in sorted(list_methods()):
        t.add_row(m, categories.get(m, ""))
    console.print(t)


# ── Benchmark config ──────────────────────────────────────────────────────

def _configure_benchmark() -> argparse.Namespace:
    console.print(f"\n  [{TEAL}]Configure Benchmark[/]\n")

    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi("Tiers", {
        "1": "fast", "2": "medium", "3": "slow", "4": "very_slow",
    }, default="1")
    tier = ",".join(tiers) if tiers else "fast"

    console.print(f"\n  [{TEAL}]B) Network size (nodes)[/]")
    nodes = _pick_one("Nodes", {"1": "20", "2": "50", "3": "100"}, default="2")

    console.print(f"\n  [{TEAL}]C) Sparsity levels[/]")
    sp = _pick_multi("Sparsities", {"1": "0.02", "2": "0.06", "3": "0.1"}, default="1,2,3")

    console.print(f"\n  [{TEAL}]D) Noise levels[/]")
    noise = _pick_multi("Noise std", {"1": "0.01", "2": "0.1", "3": "1.0"}, default="1,2,3")

    console.print(f"\n  [{TEAL}]E) Replicates[/]")
    n_ds = int(Prompt.ask(f"  [{DIM}]Number of datasets[/]", default="5"))
    timeout = int(Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="60"))

    output = Prompt.ask(f"  [{DIM}]Output file[/]", default="benchmark_results.json")

    console.print()
    return argparse.Namespace(
        tier=tier, n_nodes=int(nodes), n_samples=int(nodes) * 4,
        n_datasets=n_ds,
        sparsities=[float(s) for s in sp] if sp else [0.02, 0.06, 0.1],
        noise_levels=[float(s) for s in noise] if noise else [0.01, 0.1, 1.0],
        seed=42, timeout=timeout, output=output,
    )


# ── Benchmark runner ──────────────────────────────────────────────────────

def _run_benchmark_live(args: argparse.Namespace) -> None:
    import warnings
    warnings.simplefilter("ignore")

    from sparselink.bench.run_benchmark import TIERS, run_single
    from sparselink.bench.synthetic import generate_data, generate_network
    from sparselink import list_methods
    import sparselink.methods  # noqa: F401

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    sparsities = getattr(args, "sparsities", [0.02, 0.06, 0.1])
    noise_levels = getattr(args, "noise_levels", [0.01, 0.1, 1.0])
    topologies = ["random", "scalefree", "smallworld"]
    total = len(methods) * len(topologies) * len(sparsities) * len(noise_levels) * args.n_datasets

    console.print(f"  [{TEAL}]Methods[/]       {', '.join(methods)}")
    console.print(f"  [{TEAL}]Topologies[/]    {topologies}")
    console.print(f"  [{TEAL}]Sparsities[/]    {sparsities}")
    console.print(f"  [{TEAL}]Noise levels[/]  {noise_levels}")
    console.print(f"  [{TEAL}]Nodes[/]         {args.n_nodes}")
    console.print(f"  [{TEAL}]Datasets[/]      {args.n_datasets}")
    console.print(f"  [{TEAL}]Total runs[/]    {total}")
    console.print(f"  [{TEAL}]Timeout[/]       {args.timeout}s")
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(), TimeElapsedColumn(), console=console,
    )

    results: list[dict] = []
    rng = np.random.default_rng(args.seed)

    with progress:
        task = progress.add_task("Benchmarking", total=total)
        for topo in topologies:
            for sp in sparsities:
                for noise_std in noise_levels:
                    for ds_idx in range(args.n_datasets):
                        ds_seed = int(rng.integers(0, 2**31))
                        true_net = generate_network(args.n_nodes, topology=topo, sparsity=sp, seed=ds_seed)
                        X = generate_data(true_net, n_samples=args.n_samples, noise_std=noise_std, seed=ds_seed)
                        for method_name in methods:
                            progress.update(task, description=f"{method_name:20s} {topo}/sp={sp}/noise={noise_std}")
                            r = run_single(method_name, X, true_net, ds_idx,
                                           args.n_nodes, args.n_samples, topo, sp, noise_std, args.timeout)
                            results.append(asdict(r))
                            progress.advance(task)

    _render_results(results, f"Benchmark ({total} runs)")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── Other commands ────────────────────────────────────────────────────────

def _cmd_show(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        data = json.load(f)
    _render_results(data, f"Results from {args.file}")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    console.print(f"  [{ROSE}]Dashboard has been moved to the GeneSpider CLI.[/]")


def _cmd_status(args: argparse.Namespace) -> None:
    tree = Tree(f"[{TEAL}]sparselink[/]", guide_style="dim")

    try:
        from sparselink import list_methods
        import sparselink.methods
        methods = list_methods()
        mb = tree.add(f"Methods ({len(methods)})")
        for m in methods:
            mb.add(f"[{GREEN}]✓[/] {m}")
    except Exception as e:
        tree.add(f"[{ROSE}]✗ {e}[/]")

    accel = tree.add("Acceleration")
    try:
        import mlx.core
        accel.add(f"[{GREEN}]✓[/] MLX (Apple Silicon)")
    except ImportError:
        accel.add("[dim]○[/] MLX not available")

    deps = tree.add("Optional deps")
    for pkg, label in [("causallearn", "causal"), ("torch", "deep")]:
        try:
            __import__(pkg)
            deps.add(f"[{GREEN}]✓[/] {label}")
        except ImportError:
            deps.add(f"[dim]○[/] {label} — pip install sparselink[{label}]")

    console.print(tree)


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status",    "Show system status & available methods"),
    "2": ("methods",   "List all inference methods"),
    "3": ("infer",     "Infer network from data file"),
    "4": ("bench",     "Run synthetic benchmark"),
    "5": ("show",      "Render a previous result JSON"),
}


def _interactive() -> None:
    _print_banner()

    console.print(f"[{INDIGO}]Interactive mode[/]  [{DIM}]Ctrl+C to exit[/]\n")
    for key, (_, desc) in _MENU.items():
        console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
    console.print()

    try:
        while True:
            try:
                choice = Prompt.ask(f"[{GREEN}]sparselink ❯[/]",
                                    choices=[*_MENU, "q", "quit", "help"],
                                    show_choices=False, default="help")
            except EOFError:
                break

            if choice in ("q", "quit"):
                break
            if choice == "help":
                for key, (_, desc) in _MENU.items():
                    console.print(f"  [{INDIGO}]{key}[/]  [{DIM}]{desc}[/]")
                console.print(f"  [{INDIGO}]q[/]  [{DIM}]Quit[/]")
                continue

            cmd, _ = _MENU[choice]

            if cmd == "status":
                _cmd_status(argparse.Namespace())
            elif cmd == "methods":
                _cmd_methods(argparse.Namespace())
            elif cmd == "infer":
                path = Prompt.ask(f"  [{DIM}]Data file (.csv, .tsv, .npy)[/]")
                if not path:
                    continue
                from sparselink import list_methods
                import sparselink.methods  # noqa: F401
                method = Prompt.ask(f"  [{DIM}]Method[/]", default="genie3")
                output = Prompt.ask(f"  [{DIM}]Save adjacency to (.npy, empty=skip)[/]", default="")
                _cmd_infer(argparse.Namespace(file=path, method=method, output=output or None))
            elif cmd == "bench":
                ns = _configure_benchmark()
                _run_benchmark_live(ns)
            elif cmd == "show":
                path = Prompt.ask(f"  [{DIM}]Path to result JSON[/]")
                if path:
                    _cmd_show(argparse.Namespace(file=path))

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="sparselink", description="sparselink — network inference toolkit")
    subs = parser.add_subparsers(dest="command")

    # infer
    ip = subs.add_parser("infer", help="Infer network from data file")
    ip.add_argument("file", help="Data file (.csv, .tsv, .npy)")
    ip.add_argument("-m", "--method", default="genie3")
    ip.add_argument("-o", "--output", default=None, help="Save adjacency matrix (.npy)")

    # bench
    bp = subs.add_parser("bench", help="Run synthetic benchmark")
    bp.add_argument("--tier", default="fast")
    bp.add_argument("--n-nodes", type=int, default=50)
    bp.add_argument("--n-samples", type=int, default=200)
    bp.add_argument("--n-datasets", type=int, default=5)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--timeout", type=int, default=60)
    bp.add_argument("-o", "--output", default="benchmark_results.json")

    # show
    sp = subs.add_parser("show", help="Render previous results")
    sp.add_argument("file", help="Path to result JSON")

    # methods
    subs.add_parser("methods", help="List all inference methods")

    # status
    subs.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()
    if args.command == "infer":
        _cmd_infer(args)
    elif args.command == "bench":
        _run_benchmark_live(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "methods":
        _cmd_methods(args)
    elif args.command == "status":
        _cmd_status(args)


if __name__ == "__main__":
    main()
