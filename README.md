# sparselink

Domain-agnostic sparse network inference from tabular data. 20 methods behind a unified `fit(X) → adjacency_matrix` interface.

## Installation

```bash
pip install sparselink          # core
pip install "sparselink[causal]"  # + PC, FCI
pip install "sparselink[deep]"    # + DAG-GNN (torch)
```

## Quick Start

```python
import numpy as np
from sparselink import get_method, list_methods

print(list_methods())
# ['lasso', 'elastic_net', 'ridge', 'lsco', 'clr', 'genie3', 'tigress',
#  'partial_correlation', 'graphical_lasso', 'glasso_stars', 'neighborhood_selection',
#  'pcmci', 'granger', 'transfer_entropy', 'pc', 'fci', 'notears', 'dag_gnn', 'bdeu', 'bge']

X = np.random.randn(100, 10)
result = get_method("lasso")(alpha=0.1).fit(X)
print(result.adjacency_matrix.shape)  # (10, 10)
print(result.edge_list[:5])           # [(src, tgt, weight), ...]
```

## Interactive TUI

```bash
sparselink-tui                    # interactive menu
sparselink-tui status             # check methods, MLX, deps
sparselink-tui bench --tier fast  # run benchmark with live progress
sparselink-tui dashboard          # generate HTML dashboard
```

## Benchmarking

```python
from sparselink import get_method
from sparselink.bench import generate_network, generate_expression, evaluate

A_true = generate_network(n_genes=50, topology="scalefree", sparsity=0.06)
X = generate_expression(A_true, n_samples=150, snr=10.0)

result = get_method("genie3")().fit(X)
metrics = evaluate(A_true, result.adjacency_matrix)
print(f"AUROC={metrics.auroc:.3f}  F1={metrics.f1:.3f}  MCC={metrics.mcc:.3f}")
```

Synthetic data follows the [GeneSpider](https://bitbucket.org/sonnhammergrni/) protocol: directed networks with self-regulation, sparse identity perturbations, and chi²-scaled SNR.

## Methods

| Category | Methods |
|----------|---------|
| Regression | Lasso, Elastic Net, Ridge, LSCO, TIGRESS |
| Tree-based | GENIE3 |
| Information theory | CLR |
| Graphical models | Graphical Lasso, GLASSO+StARS, Neighborhood Selection |
| Correlation | Partial Correlation |
| Causal (time-series) | PCMCI, Granger, Transfer Entropy |
| Constraint-based | PC, FCI |
| Continuous optimization | NOTEARS, DAG-GNN |
| Bayesian | BDeu, BGe |

## Apple Silicon Acceleration

On macOS with MLX installed, matrix operations (matmul, covariance, Gram matrix) are automatically accelerated on the GPU via `sparselink.accel`.

## License

MIT
