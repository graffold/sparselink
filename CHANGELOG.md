# Changelog

All notable changes to sparselink will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed
- GeneSpider-specific synthetic data generation (`generate_expression`, SNR model)
- NestBoot bootstrap aggregation module (moved to GeneSpider CLI)
- Interactive HTML dashboard (`sparselink-dashboard` entry point)

### Changed
- Synthetic data generation now uses a generic linear model (`generate_data`)
- Benchmark parameters use `n_nodes` / `noise_std` instead of `n_genes` / `snr`
- Method docstrings use domain-agnostic terminology (features instead of genes)

### Changed
- **Breaking:** Renamed `generate_expression` to `generate_data` in `sparselink.bench.synthetic`
- **Breaking:** `BenchmarkConfig` field `n_genes` renamed to `n_nodes`, `snr` renamed to `noise_std`
- **Breaking:** CLI flags `--n-genes` and `--snr` replaced by `--n-nodes` and `--noise-std`

### Removed
- **Breaking:** NestBoot integration removed from `BenchmarkConfig` and `run_benchmark` — fields `bootstrap`, `n_bootstraps`, `n_nestings`, `fdr` and the corresponding CLI flags are no longer available; use `NestBoot` directly via the Python API instead
- CLI flags `--bootstrap`, `--n-bootstraps`, `--n-nestings`, `--fdr` removed from `sparselink bench`

### Changed (TUI)
- TUI benchmark wizard: "Network size (genes)" renamed to "Network size (nodes)"
- TUI benchmark wizard: "SNR levels" replaced by "Noise levels" (defaults: 0.01, 0.1, 1.0)

### Removed (TUI)
- NestBoot configuration removed from TUI benchmark wizard — use `NestBoot` directly via the Python API instead

### Removed (CLI)
- **Breaking:** `sparselink-dashboard` CLI entry point removed

## [1.0.0] - 2026-04-11

### Added
- Unified `fit(X) -> InferenceResult` interface via `InferenceMethod` base class
- Method registry with `@registry.register` decorator for discovery
- Common types: `AdjacencyMatrix`, `EdgeList`, `InferenceResult`, `InputData`
- 20 inference methods across 5 categories:
  - Regularization: Lasso, Elastic Net, Ridge, LSCO
  - Information-theoretic: CLR, Partial Correlation
  - Causal: PCMCI, Granger Causality, Transfer Entropy
  - Graphical models: Graphical LASSO, GLASSO+StARS, Neighborhood Selection
  - Tree/stability: GENIE3, TIGRESS
  - Constraint-based: PC, FCI
  - Continuous optimization: NOTEARS, DAG-GNN
  - Bayesian: BDeu, BGe
- `sparselink.bench` module: synthetic data, evaluation metrics, NestBoot, pipeline runner, CLI
- Full PEP 561 type annotations (py.typed)
- 94% test coverage with pytest-cov gate at 80%

[Unreleased]: https://github.com/dcolinmorgan/pyGS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dcolinmorgan/pyGS/releases/tag/v1.0.0
