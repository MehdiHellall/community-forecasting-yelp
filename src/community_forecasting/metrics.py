"""Model evaluation utilities for review-count and pulse tasks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def wape(y_true: Sequence[float] | np.ndarray, y_pred: Sequence[float] | np.ndarray) -> float:
    """Weighted absolute percentage error with explicit zero-denominator behavior."""
    truth = as_float_array(y_true)
    prediction = np.clip(as_float_array(y_pred), 0, None)
    denominator = float(np.abs(truth).sum())
    numerator = float(np.abs(truth - prediction).sum())
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def regression_metric_values(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return standard review-count regression metrics."""
    truth = as_float_array(y_true)
    prediction = np.clip(as_float_array(y_pred), 0, None)
    errors = truth - prediction
    return {
        "rows": float(len(truth)),
        "MAE": float(np.abs(errors).mean()) if len(truth) else float("nan"),
        "RMSE": float(np.sqrt(np.square(errors).mean())) if len(truth) else float("nan"),
        "WAPE": wape(truth, prediction),
    }


def safe_auc(
    metric_fn, y_true: Sequence[int] | np.ndarray, score: Sequence[float] | np.ndarray
) -> float:
    """Return NaN instead of raising when AUC is undefined for one-class labels."""
    truth = np.asarray(y_true, dtype=int)
    if len(np.unique(truth)) < 2:
        return float("nan")
    return float(metric_fn(truth, np.asarray(score, dtype=float)))


def classification_metric_values(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Return standard binary-classification metrics used by the pulse task."""
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 0, 1)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": float(recall_score(truth, prediction, zero_division=0)),
        "F1": float(f1_score(truth, prediction, zero_division=0)),
        "ROC_AUC": safe_auc(roc_auc_score, truth, score),
        "PR_AUC": safe_auc(average_precision_score, truth, score),
        "Brier": float(brier_score_loss(truth, score)),
    }


def tune_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the F1-maximizing threshold on validation labels."""
    truth = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 0, 1)
    if len(truth) == 0 or len(np.unique(truth)) < 2:
        threshold = 0.5
        return threshold, classification_metric_values(truth, score >= threshold, score)

    candidate_thresholds = list(thresholds or np.linspace(0.05, 0.95, 181))
    rows = []
    for threshold in candidate_thresholds:
        values = classification_metric_values(truth, score >= threshold, score)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": values["precision"],
                "recall": values["recall"],
                "F1": values["F1"],
            }
        )
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["F1", "precision", "recall"], ascending=[False, False, False]).iloc[0]
    threshold = float(best["threshold"])
    return threshold, classification_metric_values(truth, score >= threshold, score)


def top_k_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    k_fractions: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Evaluate pulse retrieval quality in the highest-scored rows."""
    truth = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    if len(truth) != len(score):
        raise ValueError("y_true and y_score must have the same length")

    order = np.argsort(-score, kind="mergesort")
    total_positives = int(truth.sum())
    base_rate = float(truth.mean()) if len(truth) else float("nan")
    rows = []

    for k_fraction in k_fractions:
        if not 0 < k_fraction <= 1:
            raise ValueError(f"k_fraction must be in (0, 1]: {k_fraction}")
        selected_count = max(1, int(np.ceil(len(truth) * k_fraction))) if len(truth) else 0
        selected = order[:selected_count]
        selected_positives = int(truth[selected].sum()) if selected_count else 0
        precision_at_k = selected_positives / selected_count if selected_count else float("nan")
        recall_at_k = selected_positives / total_positives if total_positives else float("nan")
        lift = precision_at_k / base_rate if base_rate else float("nan")
        rows.append(
            {
                "k_fraction": float(k_fraction),
                "selected_rows": selected_count,
                "actual_positives": total_positives,
                "selected_positives": selected_positives,
                "base_positive_rate": base_rate,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "lift_vs_base_rate": lift,
            }
        )

    return pd.DataFrame(rows)


def calibration_bins(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarize predicted probability calibration by equal-width bins."""
    truth = np.asarray(y_true, dtype=int)
    score = np.clip(np.asarray(y_score, dtype=float), 0, 1)
    if len(truth) != len(score):
        raise ValueError("y_true and y_score must have the same length")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(score, bin_edges, right=False) - 1, 0, n_bins - 1)
    brier = float(brier_score_loss(truth, score)) if len(truth) else float("nan")
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        rows.append(
            {
                "probability_bin": bin_id,
                "bin_lower": float(bin_edges[bin_id]),
                "bin_upper": float(bin_edges[bin_id + 1]),
                "rows": int(mask.sum()),
                "mean_predicted_probability": float(score[mask].mean()),
                "observed_pulse_rate": float(truth[mask].mean()),
                "brier_score": brier,
            }
        )
    return pd.DataFrame(rows)
