"""Network export utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd


def to_edge_list(adj: np.ndarray, gene_names: list[str] | None = None, threshold: float = 0.0) -> pd.DataFrame:
    """DataFrame with [source, target, weight] for edges above threshold."""
    n = adj.shape[0]
    names = gene_names or [f"V{i}" for i in range(n)]
    mask = np.abs(adj) > threshold
    np.fill_diagonal(mask, False)
    rows, cols = np.where(mask)
    return pd.DataFrame({"source": [names[i] for i in rows],
                         "target": [names[j] for j in cols],
                         "weight": adj[rows, cols]})


def to_networkx(adj: np.ndarray, gene_names: list[str] | None = None, threshold: float = 0.0):
    """Return networkx DiGraph."""
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("pip install networkx")
    df = to_edge_list(adj, gene_names, threshold)
    G = nx.DiGraph()
    n = adj.shape[0]
    G.add_nodes_from(gene_names or [f"V{i}" for i in range(n)])
    for _, r in df.iterrows():
        G.add_edge(r["source"], r["target"], weight=r["weight"])
    return G


def to_cytoscape_json(adj: np.ndarray, gene_names: list[str] | None = None, threshold: float = 0.0) -> dict:
    """Cytoscape.js JSON format."""
    n = adj.shape[0]
    names = gene_names or [f"V{i}" for i in range(n)]
    df = to_edge_list(adj, names, threshold)
    return {"nodes": [{"data": {"id": nm}} for nm in names],
            "edges": [{"data": {"source": r["source"], "target": r["target"], "weight": float(r["weight"])}} for _, r in df.iterrows()]}


def to_csv(adj: np.ndarray, path: str, gene_names: list[str] | None = None, threshold: float = 0.0) -> None:
    """Write edge list CSV."""
    to_edge_list(adj, gene_names, threshold).to_csv(path, index=False)
