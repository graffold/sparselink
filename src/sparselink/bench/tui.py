#!/usr/bin/env python3
"""sparselink TUI — graphical CLI for benchmarking network inference methods.

Usage::

    sparselink-tui                          # interactive mode
    sparselink-tui bench --tier fast        # run synthetic benchmark
    sparselink-tui bench-gs --sizes N50     # run GeneSpider benchmark
    sparselink-tui show results.json        # render previous results
    sparselink-tui dashboard -i results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
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

BANNER = [
    " ▄▄▄▄  ▄▄▄▄   ▄▄▄  ▄▄▄▄  ▄▄▄▄  ▄▄▄▄ ",
    " █▀▀▀  █▀ ▀█  █▀ █  █▀ ▀▄ █▀▀▀  █▀▀▀ ",
    " ▀▀▀█  █▀▀▀   █▀▀█  █▀▀▄  ▀▀▀█  █▀▀  ",
    " ▄▄▄█▀ █      █  █  █  █  ▄▄▄█▀ █▄▄▄ ",
]
BANNER2 = [
    " █     ▀ ▄▄▄  █  ▄ ",
    " █     █ █  █  █▄▀  ",
    " █     █ █  █  █ ▀▄ ",
    " █▄▄▄▀ █ █  █  █  █ ",
]


def _print_banner() -> None:
    for a, b in zip(BANNER, BANNER2):
        console.print(f"[{TEAL}]{a}[/][{INDIGO}]{b}[/]")
    console.print(f"  [{BOLD}]sparselink[/]  [{DIM}]network inference benchmark suite[/]")
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
    """Prompt user to pick one or more options by key. Returns selected values."""
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label} (comma-separated)[/]", default=default)
    keys = [k.strip() for k in raw.split(",")]
    return [options[k] for k in keys if k in options]


def _pick_one(label: str, options: dict[str, str], default: str = "") -> str:
    for k, v in options.items():
        console.print(f"    [{INDIGO}]{k}[/] [{DIM}]{v}[/]")
    raw = Prompt.ask(f"  [{DIM}]{label}[/]", default=default)
    return options.get(raw.strip(), options.get(default, ""))


# ── Result rendering ─────────────────────────────────────────────────────

def _render_results(data: list[dict], title: str = "Results") -> None:
    if not data:
        console.print("[dim]No results[/]")
        return

    import numpy as np
    ok = [r for r in data if not r.get("error")]
    errs = [r for r in data if r.get("error")]
    methods = sorted(set(r["method"] for r in ok))
    metrics = ["auroc", "aupr", "f1", "mcc"]

    t = Table(title=f"🧬 {title}", title_style="bold #14B8A6", border_style="dim")
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


# ── Benchmark config builder ─────────────────────────────────────────────

def _configure_synthetic() -> argparse.Namespace:
    """Interactive config builder for synthetic benchmark."""
    console.print(f"\n  [{TEAL}]Configure Synthetic Benchmark[/]\n")

    # A: Method tier
    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi("Tiers", {
        "1": "fast", "2": "medium", "3": "slow", "4": "very_slow",
    }, default="1")
    tier = ",".join(tiers) if tiers else "fast"

    # B: Network size
    console.print(f"\n  [{TEAL}]B) Network size (genes)[/]")
    genes = _pick_one("Genes", {
        "1": "20", "2": "50", "3": "100",
    }, default="2")

    # C: Sparsities
    console.print(f"\n  [{TEAL}]C) Sparsity levels[/]")
    sp_choices = _pick_multi("Sparsities", {
        "1": "0.2", "2": "0.4", "3": "0.6",
    }, default="1,2,3")

    # D: SNR levels
    console.print(f"\n  [{TEAL}]D) SNR levels[/]")
    snr_choices = _pick_multi("SNR", {
        "1": "0.1", "2": "1.0", "3": "10.0",
    }, default="1,2,3")

    # E: Datasets
    console.print(f"\n  [{TEAL}]E) Replicates[/]")
    n_datasets = int(Prompt.ask(f"  [{DIM}]Number of datasets[/]", default="5"))

    # F: Timeout
    timeout = int(Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="60"))

    # G: Output
    output = Prompt.ask(f"  [{DIM}]Output file[/]", default="benchmark_results.json")

    console.print()
    return argparse.Namespace(
        tier=tier,
        n_genes=int(genes),
        n_samples=int(genes) * 4,
        n_datasets=n_datasets,
        sparsities=[float(s) for s in sp_choices] if sp_choices else [0.2, 0.4, 0.6],
        snr_levels=[float(s) for s in snr_choices] if snr_choices else [0.1, 1.0, 10.0],
        seed=42,
        timeout=timeout,
        output=output,
    )


def _configure_genespider() -> argparse.Namespace:
    """Interactive config builder for GeneSpider benchmark."""
    console.print(f"\n  [{TEAL}]Configure GeneSpider Benchmark[/]\n")

    # A: Method tier
    console.print(f"  [{TEAL}]A) Method tier[/]")
    tiers = _pick_multi("Tiers", {
        "1": "fast", "2": "medium", "3": "slow",
    }, default="1")
    tier = ",".join(tiers) if tiers else "fast"

    # B: Network sizes
    console.print(f"\n  [{TEAL}]B) Network sizes[/]")
    sizes = _pick_multi("Sizes", {
        "1": "N10", "2": "N50", "3": "N100",
    }, default="2")
    sizes_str = ",".join(sizes) if sizes else "N50"

    # C: Max datasets
    console.print(f"\n  [{TEAL}]C) Max datasets per size[/]")
    max_ds = int(Prompt.ask(f"  [{DIM}]0 = all[/]", default="0"))

    # D: Timeout
    timeout = int(Prompt.ask(f"\n  [{DIM}]Timeout per method (seconds)[/]", default="120"))

    # E: Output
    output = Prompt.ask(f"  [{DIM}]Output file[/]", default="benchmark_genespider.json")

    console.print()
    return argparse.Namespace(
        tier=tier, sizes=sizes_str, max_datasets=max_ds,
        timeout=timeout, output=output,
    )


# ── Live benchmark runners ───────────────────────────────────────────────

def _run_benchmark_live(args: argparse.Namespace) -> None:
    """Run synthetic benchmark with live progress."""
    import warnings
    warnings.simplefilter("ignore")

    from sparselink.bench.run_benchmark import (
        TIERS, RunResult, run_single,
    )
    from sparselink.bench.synthetic import generate_expression, generate_network
    from sparselink import list_methods
    import sparselink.methods  # noqa: F401
    import numpy as np

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    sparsities = getattr(args, "sparsities", [0.2, 0.4, 0.6])
    snr_levels = getattr(args, "snr_levels", [0.1, 1.0, 10.0])
    topologies = ["random", "scalefree", "smallworld"]

    total = len(methods) * len(topologies) * len(sparsities) * len(snr_levels) * args.n_datasets

    console.print(f"  [{TEAL}]Methods[/]     {', '.join(methods)}")
    console.print(f"  [{TEAL}]Topologies[/]  {topologies}")
    console.print(f"  [{TEAL}]Sparsities[/]  {sparsities}")
    console.print(f"  [{TEAL}]SNR levels[/]  {snr_levels}")
    console.print(f"  [{TEAL}]Genes[/]       {args.n_genes}")
    console.print(f"  [{TEAL}]Datasets[/]    {args.n_datasets}")
    console.print(f"  [{TEAL}]Total runs[/]  {total}")
    console.print(f"  [{TEAL}]Timeout[/]     {args.timeout}s")
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []
    rng = np.random.default_rng(args.seed)

    with progress:
        task = progress.add_task("Benchmarking", total=total)
        for topo in topologies:
            for sp in sparsities:
                for snr in snr_levels:
                    for ds_idx in range(args.n_datasets):
                        ds_seed = int(rng.integers(0, 2**31))
                        true_net = generate_network(args.n_genes, topology=topo, sparsity=sp, seed=ds_seed)
                        X = generate_expression(true_net, n_samples=args.n_samples, snr=snr, seed=ds_seed)
                        for method_name in methods:
                            progress.update(task, description=f"{method_name:20s} {topo}/sp={sp}/SNR={snr}")
                            r = run_single(method_name, X, true_net, ds_idx,
                                           args.n_genes, args.n_samples, topo, sp, snr, args.timeout)
                            results.append(asdict(r))
                            progress.advance(task)

    _render_results(results, f"Synthetic Benchmark ({total} runs)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


def _run_genespider_live(args: argparse.Namespace) -> None:
    """Run GeneSpider benchmark with live progress (requires pyGS)."""
    try:
        from benchmark_genespider import (
            TIERS, _list_datasets, load_dataset, run_single,
        )
    except ImportError:
        console.print(f"  [{ROSE}]✗ GeneSpider benchmark requires pyGS.[/]")
        console.print(f"  [{DIM}]Install pyGS or run from the pyGS repo directory.[/]")
        return

    import warnings
    warnings.simplefilter("ignore")

    from sparselink import list_methods
    import sparselink.methods  # noqa: F401

    selected_tiers = [t.strip() for t in args.tier.split(",")]
    methods: list[str] = []
    for t in selected_tiers:
        methods.extend(TIERS.get(t, []))
    registered = set(list_methods())
    methods = [m for m in methods if m in registered]

    sizes = [s.strip() for s in args.sizes.split(",")]

    console.print(f"  [{TEAL}]Methods[/]  {', '.join(methods)}")
    console.print(f"  [{TEAL}]Sizes[/]    {sizes}")
    console.print(f"  [{TEAL}]Timeout[/]  {args.timeout}s")
    console.print()

    progress = Progress(
        SpinnerColumn(style=TEAL),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, complete_style=TEAL, finished_style=GREEN),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    results: list[dict] = []

    for size in sizes:
        console.print(f"  [{INDIGO}]Fetching {size} datasets...[/]")
        datasets = _list_datasets(size)
        if args.max_datasets > 0:
            datasets = datasets[:args.max_datasets]

        total = len(datasets) * len(methods)
        console.print(f"  [{DIM}]{len(datasets)} datasets × {len(methods)} methods = {total} runs[/]")

        with progress:
            task = progress.add_task(f"{size}", total=total)
            for ds_meta in datasets:
                try:
                    X, P, A_true, topology, net_name = load_dataset(ds_meta, size)
                except Exception:
                    progress.advance(task, len(methods))
                    continue
                for method_name in methods:
                    snr = ds_meta["snr"]
                    progress.update(task, description=f"{method_name:20s} {topology}/SNR={snr}")
                    r = run_single(method_name, X, A_true, ds_meta, topology, net_name, args.timeout, P)
                    results.append(asdict(r))
                    progress.advance(task)

    _render_results(results, f"GeneSpider Benchmark ({len(results)} runs)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n  [{DIM}]Results saved to {args.output}[/]")


# ── Commands ──────────────────────────────────────────────────────────────

def _cmd_show(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        data = json.load(f)
    _render_results(data, f"Results from {args.file}")


def _cmd_dashboard(args: argparse.Namespace) -> None:
    from sparselink.bench.dashboard import main as dash_main
    dash_main(["-i", args.input, "-o", args.output])
    console.print(f"  [{GREEN}]✓[/] Dashboard written to {args.output}")
    if not args.no_open:
        import subprocess
        subprocess.run(["open", args.output], check=False)


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
        tree.add(f"[{ROSE}]✗ sparselink not importable: {e}[/]")

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

    cache = Path(".gs_cache")
    if cache.exists():
        n = len(list(cache.glob("*.json")))
        tree.add(f"[{GREEN}]✓[/] GeneSpider cache: {n} files")
    else:
        tree.add("[dim]○[/] No GeneSpider cache")

    console.print(tree)


# ── Interactive mode ──────────────────────────────────────────────────────

_MENU = {
    "1": ("status",    "Show system status & available methods"),
    "2": ("bench",     "Run synthetic benchmark"),
    "3": ("bench-gs",  "Run GeneSpider benchmark"),
    "4": ("dashboard", "Generate interactive HTML dashboard"),
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

            elif cmd == "bench":
                ns = _configure_synthetic()
                _run_benchmark_live(ns)

            elif cmd == "bench-gs":
                ns = _configure_genespider()
                _run_genespider_live(ns)

            elif cmd == "dashboard":
                inp = Prompt.ask(f"  [{DIM}]Input JSON[/]", default="benchmark_results.json")
                out = Prompt.ask(f"  [{DIM}]Output HTML[/]", default="benchmark_dashboard.html")
                _cmd_dashboard(argparse.Namespace(input=inp, output=out, no_open=False))

            elif cmd == "show":
                path = Prompt.ask(f"  [{DIM}]Path to result JSON[/]")
                if path:
                    _cmd_show(argparse.Namespace(file=path))

            console.print()

    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]bye[/]")


# ── CLI entrypoint ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="sparselink-tui", description="sparselink graphical CLI")
    subs = parser.add_subparsers(dest="command")

    bp = subs.add_parser("bench", help="Run synthetic benchmark")
    bp.add_argument("--tier", default="fast")
    bp.add_argument("--n-genes", type=int, default=50)
    bp.add_argument("--n-samples", type=int, default=200)
    bp.add_argument("--n-datasets", type=int, default=5)
    bp.add_argument("--seed", type=int, default=42)
    bp.add_argument("--timeout", type=int, default=60)
    bp.add_argument("-o", "--output", default="benchmark_results.json")

    gp = subs.add_parser("bench-gs", help="Run GeneSpider benchmark")
    gp.add_argument("--tier", default="fast")
    gp.add_argument("--sizes", default="N50")
    gp.add_argument("--max-datasets", type=int, default=0)
    gp.add_argument("--timeout", type=int, default=120)
    gp.add_argument("-o", "--output", default="benchmark_genespider.json")

    sp = subs.add_parser("show", help="Render previous results")
    sp.add_argument("file", help="Path to result JSON")

    dp = subs.add_parser("dashboard", help="Generate interactive HTML dashboard")
    dp.add_argument("-i", "--input", default="benchmark_results.json")
    dp.add_argument("-o", "--output", default="benchmark_dashboard.html")
    dp.add_argument("--no-open", action="store_true")

    subs.add_parser("status", help="Show system status")

    args = parser.parse_args()

    if not args.command:
        _interactive()
        return

    _print_banner()
    if args.command == "bench":
        _run_benchmark_live(args)
    elif args.command == "bench-gs":
        _run_genespider_live(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "status":
        _cmd_status(args)


if __name__ == "__main__":
    main()
