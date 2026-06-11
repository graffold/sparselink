"""sparselink - Domain-agnostic sparse network inference from tabular data."""

import sparselink.methods  # noqa: F401 — trigger registration
from sparselink.base import InferenceMethod
from sparselink.registry import get_method, list_methods, registry
from sparselink.types import AdjacencyMatrix, EdgeList, InferenceResult

__version__ = "1.2.0"

__all__ = [
    "AdjacencyMatrix",
    "EdgeList",
    "InferenceResult",
    "InferenceMethod",
    "registry",
    "get_method",
    "list_methods",
]
