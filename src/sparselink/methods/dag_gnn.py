"""DAG-GNN: DAG Structure Learning with Graph Neural Networks (requires torch)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sparselink.base import InferenceMethod
from sparselink.registry import registry
from sparselink.types import InferenceResult, InputData


@registry.register
class DAGGNNMethod(InferenceMethod):
    """DAG-GNN structure learning using a variational autoencoder with DAG constraint.

    Simplified implementation: learns adjacency via gradient descent with
    acyclicity penalty using matrix exponential trace constraint.

    Requires: pip install sparselink[deep]
    """

    name = "dag_gnn"

    def __init__(
        self,
        hidden_dim: int = 16,
        epochs: int = 200,
        lr: float = 3e-3,
        lambda1: float = 0.01,
        w_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            epochs=epochs,
            lr=lr,
            lambda1=lambda1,
            w_threshold=w_threshold,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.lambda1 = lambda1
        self.w_threshold = w_threshold

    def fit(self, X: InputData, y: InputData | None = None) -> InferenceResult:
        """Run DAG-GNN.

        Args:
            X: Data matrix (samples x features).
            y: Ignored.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "DAG-GNN requires PyTorch. Install with: pip install sparselink[deep]"
            ) from e

        X_arr = self._to_array(X)
        X_arr = X_arr - X_arr.mean(axis=0)
        n_samples, d = X_arr.shape

        X_t = torch.tensor(X_arr, dtype=torch.float32)

        # Learnable adjacency (d x d)
        A = nn.Parameter(torch.zeros(d, d))
        encoder = nn.Sequential(nn.Linear(d, self.hidden_dim), nn.ReLU())
        decoder = nn.Sequential(nn.Linear(self.hidden_dim, d))

        params = list(encoder.parameters()) + list(decoder.parameters()) + [A]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        rho = 1.0
        alpha = 0.0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # Graph-weighted input: X @ A gives neighbor aggregation
            X_in = X_t @ A
            Z = encoder(X_in)
            X_hat = decoder(Z)

            recon_loss = torch.mean((X_t - X_hat) ** 2)
            l1_loss = self.lambda1 * torch.sum(torch.abs(A))

            # Acyclicity: tr(e^(A◦A)) - d
            M = A * A
            h = torch.trace(torch.linalg.matrix_exp(M)) - d

            loss = recon_loss + l1_loss + alpha * h + 0.5 * rho * h * h
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                A.fill_diagonal_(0.0)

            # Update Lagrangian
            if (epoch + 1) % 50 == 0:
                h_val = h.item()
                alpha += rho * h_val
                if h_val > 0.25:
                    rho *= 2.0

        W = A.detach().numpy()
        W[np.abs(W) < self.w_threshold] = 0.0
        np.fill_diagonal(W, 0.0)
        return InferenceResult(adjacency_matrix=np.abs(W))
