"""Validation checks for committed portfolio artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str


REQUIRED_OUTPUT_COLUMNS = {
    "model_comparison_summary.csv": {
        "split",
        "best_review_count_model",
        "best_review_count_WAPE",
        "best_pulse_model",
        "best_pulse_F1",
        "best_top10_precision",
        "modality_takeaway",
    },
    "forecasting_metrics.csv": {"split", "task", "model", "rows", "MAE", "RMSE", "WAPE"},
    "attention_pulse_metrics.csv": {
        "split",
        "task",
        "model",
        "rows",
        "positive_rate",
        "decision_threshold",
        "F1",
        "PR_AUC",
        "Brier",
    },
    "attention_pulse_topk_metrics.csv": {
        "split",
        "task",
        "model",
        "k_fraction",
        "precision_at_k",
        "lift_vs_base_rate",
    },
    "attention_pulse_calibration.csv": {
        "split",
        "model",
        "probability_bin",
        "mean_predicted_probability",
        "observed_pulse_rate",
        "brier_score",
    },
    "attention_pulse_case_studies.csv": {
        "case_type",
        "model",
        "business_id",
        "name",
        "target_month_str",
        "interpretation_note",
    },
    "nlp_tfidf_terms.csv": {"term", "feature"},
    "pulse_predecessor_analysis.csv": {
        "feature",
        "non_pulse_mean",
        "pulse_mean",
        "absolute_difference",
    },
}

REQUIRED_FIGURE_DIRS = [
    "eda",
    "feature_engineering",
    "models",
    "sna",
]


def validate_outputs(outputs_dir: str | Path) -> list[CheckResult]:
    """Validate that committed summary outputs are present and readable."""
    root = Path(outputs_dir)
    results: list[CheckResult] = []
    for filename, required_columns in REQUIRED_OUTPUT_COLUMNS.items():
        path = root / filename
        if not path.exists():
            results.append(CheckResult(filename, False, "missing"))
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - pandas gives many parser errors
            results.append(CheckResult(filename, False, f"unreadable CSV: {exc}"))
            continue
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            results.append(CheckResult(filename, False, f"missing columns: {missing}"))
            continue
        if frame.empty:
            results.append(CheckResult(filename, False, "contains no rows"))
            continue
        results.append(CheckResult(filename, True, f"{len(frame):,} rows"))

    figures_root = root / "figures"
    for directory in REQUIRED_FIGURE_DIRS:
        path = figures_root / directory
        png_count = len(list(path.glob("*.png"))) if path.exists() else 0
        results.append(
            CheckResult(
                f"figures/{directory}",
                png_count > 0,
                f"{png_count} PNG files" if png_count > 0 else "missing PNG files",
            )
        )
    return results


def load_result_summary(outputs_dir: str | Path) -> dict[str, str]:
    """Load the single-row project result summary used by the README and CLI."""
    path = Path(outputs_dir) / "model_comparison_summary.csv"
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"{path} should contain exactly one summary row")
    return {key: str(value) for key, value in frame.iloc[0].to_dict().items()}
