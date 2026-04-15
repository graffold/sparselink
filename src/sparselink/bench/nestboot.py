"""NestBoot: bootstrap aggregation with FDR control for network inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sparselink.base import InferenceMethod
from sparselink.types import InputData


@dataclass
class NestBootResult:
    """Result of NestBoot aggregation."""

    adjacency: npt.NDArray[np.floating]
    frequency: npt.NDArray[np.floating]
    fdr_threshold: float


class NestBoot:
    """Bootstrap aggregation with FDR control for network inference.

    Runs an inference method on bootstrap resamples, aggregates edge
    frequencies, and applies FDR control using a null (shuffled) distribution.
    """

    def __init__(
        self,
        n_bootstraps: int = 50,
        n_nestings: int = 5,
        fdr: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.n_bootstraps = n_bootstraps
        self.n_nestings = n_nestings
        self.fdr = fdr
        self.seed = seed

    def run(
        self,
        method: InferenceMethod,
        X: InputData,
    ) -> NestBootResult:
        """Run NestBoot on data with given method.

        Args:
            method: An InferenceMethod instance.
            X: Input data (samples x features).

        Returns:
            NestBootResult with aggregated adjacency and FDR threshold.
        """
        data = np.asarray(X)
        n_samples, n_features = data.shape
        rng = np.random.default_rng(self.seed)

        # Accumulate edge frequencies from real data
        freq_real = np.zeros((n_features, n_features))
        for _ in range(self.n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            result = method.fit(data[idx])
            freq_real += (result.adjacency_matrix != 0).astype(float)
        freq_real /= self.n_bootstraps

        # Null distribution: shuffled columns independently
        freq_null = np.zeros((n_features, n_features))
        for _ in range(self.n_nestings):
            shuffled = data.copy()
            for col in range(n_features):
                rng.shuffle(shuffled[:, col])
            for _ in range(self.n_bootstraps // self.n_nestings or 1):
                idx = rng.choice(n_samples, size=n_samples, replace=True)
                result = method.fit(shuffled[idx])
                freq_null += (result.adjacency_matrix != 0).astype(float)
        total_null = self.n_nestings * max(1, self.n_bootstraps // self.n_nestings)
        freq_null /= total_null

        # FDR control: find threshold where FDR <= target
        threshold = self._fdr_threshold(freq_real, freq_null)

        adjacency = np.where(freq_real >= threshold, freq_real, 0.0)

        return NestBootResult(
            adjacency=adjacency,
            frequency=freq_real,
            fdr_threshold=threshold,
        )

    def _fdr_threshold(
        self,
        freq_real: npt.NDArray[np.floating],
        freq_null: npt.NDArray[np.floating],
    ) -> float:
        """Find frequency threshold achieving target FDR."""
        mask = ~np.eye(freq_real.shape[0], dtype=bool)
        for t in np.linspace(1.0, 0.0, 101):
            n_real = np.sum(freq_real[mask] >= t)
            n_null = np.sum(freq_null[mask] >= t)
            if n_real > 0:
                estimated_fdr = n_null / n_real
                if estimated_fdr <= self.fdr:
                    return float(t)
        return 1.0
