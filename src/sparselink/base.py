"""Base class for all inference methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from sparselink.types import InferenceResult, InputData


class InferenceMethod(ABC):
    """Abstract base for all network inference algorithms.

    Subclasses must implement `fit` and set `name` as a class attribute.
    """

    name: str = ""

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs

    @abstractmethod
    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Infer network structure from data.

        Args:
            X: Input data matrix (samples x features) as numpy array or DataFrame.
            y: Optional target/perturbation matrix for supervised methods.

        Returns:
            InferenceResult containing the inferred adjacency matrix.
        """
        ...

    def _to_array(self, data: InputData) -> np.ndarray:  # noqa: E501
        """Convert input to numpy array."""
        if isinstance(data, pd.DataFrame):
            return np.asarray(data.values)
        return np.asarray(data)
