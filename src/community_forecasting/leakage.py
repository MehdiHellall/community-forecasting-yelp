"""Leakage guardrails for the forecasting workflow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import nbformat
import pandas as pd

FORBIDDEN_SNAPSHOT_FEATURES = frozenset({"business_review_count", "business_stars"})
FORBIDDEN_NOTEBOOK_OUTPUT_TERMS = frozenset(
    {
        "<td>business_review_count</td>",
        "<td>business_stars</td>",
        "`business_review_count` rises",
        "`business_stars` rises",
    }
)
DEFAULT_NON_FEATURE_COLUMNS = frozenset(
    {
        "business_id",
        "name",
        "feature_month_str",
        "target_month_str",
        "target_next_month_reviews",
        "attention_pulse",
        "prediction",
        "probability",
        "decision_threshold",
        "pulse_baseline_reviews",
        "pulse_relative_lift",
        "split",
        "model",
    }
)


@dataclass(frozen=True)
class LeakageFinding:
    severity: str
    message: str


def find_forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    """Return model feature columns known to leak from Yelp snapshot metadata."""
    return sorted(FORBIDDEN_SNAPSHOT_FEATURES.intersection(columns))


def select_safe_feature_columns(
    columns: Iterable[str],
    *,
    extra_excluded: Iterable[str] = (),
) -> list[str]:
    """Select candidate model features while excluding IDs, targets, predictions, and leaks."""
    excluded = DEFAULT_NON_FEATURE_COLUMNS.union(extra_excluded).union(FORBIDDEN_SNAPSHOT_FEATURES)
    return [
        column for column in columns if column not in excluded and not column.startswith("future_")
    ]


def check_sna_summary(
    summary: dict,
    *,
    max_review_window_end: str = "2017-12",
) -> list[LeakageFinding]:
    """Validate that persisted SNA metadata is train-window only."""
    findings: list[LeakageFinding] = []
    review_window = summary.get("review_window") or {}
    review_window_end = review_window.get("end") or summary.get("review_window_end")
    if review_window_end is None:
        findings.append(
            LeakageFinding("error", "SNA summary is missing review_window.end metadata.")
        )
        return findings
    if pd.Period(str(review_window_end), freq="M") > pd.Period(max_review_window_end, freq="M"):
        findings.append(
            LeakageFinding(
                "error",
                f"SNA review window ends at {review_window_end}; expected no later than "
                f"{max_review_window_end}.",
            )
        )
    return findings


def check_modeling_frame(
    frame: pd.DataFrame,
    *,
    feature_month_column: str = "feature_month_str",
    target_month_column: str = "target_month_str",
    max_target_month: str = "2019-12",
) -> list[LeakageFinding]:
    """Check dataframe-level temporal and feature leakage risks."""
    findings: list[LeakageFinding] = []
    forbidden = find_forbidden_feature_columns(frame.columns)
    if forbidden:
        findings.append(
            LeakageFinding(
                "error",
                f"Forbidden snapshot-derived model features are present: {', '.join(forbidden)}",
            )
        )

    missing = [
        column
        for column in [feature_month_column, target_month_column]
        if column not in frame.columns
    ]
    if missing:
        findings.append(LeakageFinding("error", f"Missing temporal columns: {', '.join(missing)}"))
        return findings

    feature_months = pd.PeriodIndex(frame[feature_month_column].astype(str), freq="M")
    target_months = pd.PeriodIndex(frame[target_month_column].astype(str), freq="M")
    if (feature_months >= target_months).any():
        findings.append(
            LeakageFinding("error", "Feature months must be strictly earlier than target months.")
        )
    if (target_months > pd.Period(max_target_month, freq="M")).any():
        findings.append(
            LeakageFinding(
                "error", f"Target months exceed the locked test window ending {max_target_month}."
            )
        )
    return findings


def scan_notebooks_for_static_leakage(notebooks_dir: str | Path) -> list[LeakageFinding]:
    """Scan notebooks for feature-list usage of known snapshot-derived columns.

    The scan is intentionally narrow: it flags business snapshot fields when they
    appear inside notebook feature-set definitions, while allowing docs to mention
    the rejected fields.
    """
    findings: list[LeakageFinding] = []
    notebook_dir = Path(notebooks_dir)
    for path in sorted(notebook_dir.glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        rendered_text = path.read_text(encoding="utf-8")
        forbidden_output_terms = sorted(
            term for term in FORBIDDEN_NOTEBOOK_OUTPUT_TERMS if term in rendered_text
        )
        if forbidden_output_terms:
            findings.append(
                LeakageFinding(
                    "error",
                    f"{path.name} contains stale notebook output or interpretation for "
                    f"forbidden snapshot features.",
                )
            )
        code_sources = [
            cell.get("source", "") for cell in notebook.cells if cell.get("cell_type") == "code"
        ]
        for body in _business_feature_blocks(code_sources):
            forbidden = [
                feature for feature in FORBIDDEN_SNAPSHOT_FEATURES if f'"{feature}"' in body
            ]
            if not forbidden:
                continue
            findings.append(
                LeakageFinding(
                    "error",
                    f"{path.name} includes forbidden snapshot features in a feature set: "
                    f"{', '.join(sorted(forbidden))}",
                )
            )
    return findings


def _business_feature_blocks(code_sources: Iterable[str]) -> list[str]:
    blocks: list[str] = []
    for source in code_sources:
        in_block = False
        block_lines: list[str] = []
        for line in source.splitlines():
            if line.strip() == "business_features = [":
                in_block = True
                block_lines = []
                continue
            if in_block and line.strip() == "]":
                blocks.append("\n".join(block_lines))
                in_block = False
                continue
            if in_block:
                block_lines.append(line)
    return blocks
