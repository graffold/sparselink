# sparselink

Domain-agnostic sparse network inference from tabular data.

sparselink provides a unified Python interface for 20+ network inference algorithms — from LASSO and graphical models to causal discovery and deep learning approaches. All methods share a single `fit(X) -> InferenceResult` API, making it easy to swap algorithms, benchmark them on synthetic data, and compare results.

## Installation

```bash
pip install sparselink
```

Optional extras:

```bash
pip install sparselink[causal]   # PC, FCI (causal-learn)
pip install sparselink[deep]     # DAG-GNN (PyTorch)
pip install sparselink[dev]      # pytest, mypy, ruff
```

## Quick Start

```python
import numpy as np
from sparselink import get_method

X = np.random.default_rng(42).standard_normal((100, 20))

method = get_method("lasso")
result = method(alpha=0.01).fit(X)

print(result.adjacency_matrix.shape)  # (20, 20)
print(result.edge_list[:5])           # [(src, tgt, weight), ...]
```

## Available Methods

| Category | Methods |
|---|---|
| Regularization | Lasso, Elastic Net, Ridge, LSCO |
| Information-theoretic | CLR, Partial Correlation |
| Graphical models | Graphical LASSO, GLASSO+StARS, Neighborhood Selection |
| Tree / stability | GENIE3, TIGRESS |
| Causal (time-series) | PCMCI, Granger Causality, Transfer Entropy |
| Constraint-based | PC, FCI |
| Continuous optimization | NOTEARS, DAG-GNN |
| Bayesian | BDeu, BGe |

```python
from sparselink import list_methods
print(list_methods())
```

## Method Registry

Methods are registered via the `@registry.register` decorator and discovered automatically at import time. Use `get_method(name)` to retrieve any method by its string name — no direct imports needed in application code.

## Benchmarking

sparselink includes a benchmarking suite with synthetic network/data generation and evaluation metrics for comparing algorithm performance.

### Python API

```python
from sparselink.bench import generate_network, generate_data, evaluate
from sparselink import get_method

net = generate_network(20, topology="scalefree", sparsity=0.1, seed=0)
X = generate_data(net, n_samples=100, noise_std=0.1, seed=0)

result = get_method("genie3")().fit(X)
metrics = evaluate(net, result.adjacency_matrix)
print(f"AUROC={metrics.auroc:.3f}  F1={metrics.f1:.3f}")
```

### CLI

```bash
# Batch benchmark
sparselink-bench --methods lasso elastic_net genie3 --n-datasets 10

# Exhaustive benchmark with tiers and timeout
sparselink-benchmark --tier fast,medium --timeout 60

# Interactive TUI
sparselink-tui
```

## Evaluation Metrics

- AUROC, AUPR (threshold-free)
- Precision, Recall, F1, FDR, MCC (best over threshold sweep)
- R² (edge weight accuracy)

## Project Structure

```
src/sparselink/
├── base.py          # InferenceMethod ABC
├── types.py         # AdjacencyMatrix, EdgeList, InferenceResult
├── registry.py      # @registry.register decorator + discovery
├── accel.py         # Optional MLX acceleration (Apple Silicon)
├── methods/         # One module per algorithm
└── bench/           # Synthetic data, metrics, runner, CLI
```

## Development

```bash
pip install -e ".[dev]"

pytest                       # tests + coverage (80% gate)
mypy src/                    # strict type checking
ruff check src/ tests/       # lint
ruff format src/ tests/      # format
```

## License

MIT
