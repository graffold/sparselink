"""Method registry for discovery and enumeration of inference algorithms."""

from __future__ import annotations

from sparselink.base import InferenceMethod


class Registry:
    """Registry for inference method classes."""

    def __init__(self) -> None:
        self._methods: dict[str, type[InferenceMethod]] = {}

    def register(self, cls: type[InferenceMethod]) -> type[InferenceMethod]:
        """Register a method class. Use as decorator."""
        name = cls.name or cls.__name__.lower()
        self._methods[name] = cls
        return cls

    def get(self, name: str) -> type[InferenceMethod]:
        """Get a registered method by name."""
        if name not in self._methods:
            raise KeyError(f"Unknown method '{name}'. Available: {list(self._methods)}")
        return self._methods[name]

    def list(self) -> list[str]:
        """List all registered method names."""
        return list(self._methods.keys())


registry = Registry()


def get_method(name: str) -> type[InferenceMethod]:
    """Get a registered inference method by name."""
    return registry.get(name)


def list_methods() -> list[str]:
    """List all available inference method names."""
    return registry.list()
