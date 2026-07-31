from __future__ import annotations

import math

import numpy as np
import pytest

from community_forecasting.metrics import (
    calibration_bins,
    classification_metric_values,
    regression_metric_values,
    top_k_metrics,
    tune_threshold,
    wape,
)


def test_wape_matches_known_value_and_clips_negative_predictions():
    assert wape([10, 20, 0], [8, 25, -3]) == pytest.approx(7 / 30)


def test_wape_handles_zero_denominator_explicitly():
    assert wape([0, 0], [0, 0]) == 0.0
    assert math.isinf(wape([0, 0], [1, 0]))


def test_regression_metrics_clip_negative_predictions():
    metrics = regression_metric_values([0, 1, 3, 6], [0, 2, -1, 7])

    assert metrics["rows"] == 4
    assert metrics["MAE"] == pytest.approx(1.25)
    assert metrics["RMSE"] == pytest.approx(1.6583123951777)
    assert metrics["WAPE"] == pytest.approx(0.5)


def test_classification_metrics_match_known_values():
    metrics = classification_metric_values([0, 1, 1, 0], [0, 1, 0, 1], [0.1, 0.9, 0.4, 0.7])

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["F1"] == pytest.approx(0.5)
    assert metrics["Brier"] == pytest.approx(np.mean([0.01, 0.01, 0.36, 0.49]))


def test_tune_threshold_uses_f1_then_precision_then_recall():
    threshold, metrics = tune_threshold([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.7], thresholds=[0.3, 0.5])

    assert threshold == 0.3
    assert metrics["F1"] == pytest.approx(0.8)


def test_top_k_metrics_rank_highest_scores():
    metrics = top_k_metrics([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], k_fractions=[0.5])

    row = metrics.iloc[0]
    assert row["selected_rows"] == 2
    assert row["selected_positives"] == 1
    assert row["precision_at_k"] == pytest.approx(0.5)
    assert row["recall_at_k"] == pytest.approx(0.5)
    assert row["lift_vs_base_rate"] == pytest.approx(1.0)


def test_calibration_bins_summarize_observed_rates():
    bins = calibration_bins([0, 1, 1, 0], [0.05, 0.25, 0.75, 0.95], n_bins=2)

    assert bins["probability_bin"].tolist() == [0, 1]
    assert bins["rows"].tolist() == [2, 2]
    assert bins["observed_pulse_rate"].tolist() == [0.5, 0.5]


def test_top_k_metrics_rounds_small_samples_up_to_one_row():
    metrics = top_k_metrics([0, 1, 0], [0.1, 0.9, 0.2], k_fractions=[0.1])

    assert metrics.iloc[0]["selected_rows"] == 1
