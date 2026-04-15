"""Evaluation metrics for network inference benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""

    auroc: float
    aupr: float
    precision: float
    recall: float
    f1: float
    fdr: float
    mcc: float
    r2: float


def evaluate(
    true_network: npt.NDArray[np.floating],
    predicted: npt.NDArray[np.floating],
    threshold: float | None = None,
    n_thresholds: int = 30,
) -> MetricsResult:
    """Compute evaluation metrics comparing predicted to true network.

    Sweeps over ``n_thresholds`` values and reports the best F1/MCC found,
    matching the GeneSpider evaluation protocol. AUROC and AUPR are computed
    from continuous scores (threshold-free).

    Args:
        true_network: Ground-truth adjacency matrix (n x n).
        predicted: Predicted adjacency/weight matrix (n x n).
        threshold: If provided, evaluate at this single threshold only.
        n_thresholds: Number of thresholds to sweep (default 30).

    Returns:
        MetricsResult with best metrics across the threshold sweep.
    """
    n = true_network.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y_true = (true_network[mask] != 0).astype(int)
    y_scores = np.abs(predicted[mask])

    # AUROC and AUPR are threshold-free
    try:
        auroc = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        auroc = 0.5
    try:
        aupr = float(average_precision_score(y_true, y_scores))
    except ValueError:
        aupr = 0.0

    # R² between true edge weights and predicted weights
    true_flat = np.abs(true_network[mask]).astype(float)
    pred_flat = y_scores.astype(float)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Sweep thresholds for binary metrics (GeneSpider protocol)
    if threshold is not None:
        thresholds = [threshold]
    else:
        nonzero = y_scores[y_scores > 0]
        if len(nonzero) == 0:
            return MetricsResult(auroc=auroc, aupr=aupr, precision=0, recall=0,
                                 f1=0, fdr=1, mcc=0, r2=r2)
        thresholds = np.linspace(nonzero.min(), nonzero.max(), n_thresholds).tolist()

    best_f1, best_mcc = -1.0, -2.0
    best_prec, best_rec, best_fdr = 0.0, 0.0, 1.0

    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        if not np.any(y_pred):
            continue
        f1_val = float(f1_score(y_true, y_pred, zero_division=0))
        mcc_val = float(matthews_corrcoef(y_true, y_pred))
        if f1_val > best_f1:
            best_f1 = f1_val
            best_prec = float(precision_score(y_true, y_pred, zero_division=0))
            best_rec = float(recall_score(y_true, y_pred, zero_division=0))
            best_fdr = 1.0 - best_prec
        if mcc_val > best_mcc:
            best_mcc = mcc_val

    if best_f1 < 0:
        best_f1 = 0.0
    if best_mcc < -1:
        best_mcc = 0.0

    return MetricsResult(
        auroc=auroc, aupr=aupr, precision=best_prec, recall=best_rec,
        f1=best_f1, fdr=best_fdr, mcc=best_mcc, r2=r2,
    )
