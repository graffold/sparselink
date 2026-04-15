"""Common types for sparselink."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

# Core type aliases
AdjacencyMatrix = npt.NDArray[np.floating]  # 2D (n x n) array
EdgeList = list[tuple[int, int, float]]  # [(source, target, weight), ...]


@dataclass
class InferenceResult:
    """Result container for network inference methods."""

    adjacency_matrix: AdjacencyMatrix
    edge_list: EdgeList = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        if not self.edge_list:
            self.edge_list = self._matrix_to_edges(self.adjacency_matrix)

    @staticmethod
    def _matrix_to_edges(mat: AdjacencyMatrix) -> EdgeList:
        rows, cols = np.nonzero(mat)
        return [(int(r), int(c), float(mat[r, c])) for r, c in zip(rows, cols)]


# Input type accepted by all methods
InputData = npt.NDArray[np.floating] | pd.DataFrame
