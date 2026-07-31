from __future__ import annotations

import pandas as pd

from community_forecasting.leakage import (
    check_modeling_frame,
    check_sna_summary,
    find_forbidden_feature_columns,
    select_safe_feature_columns,
)


def test_forbidden_snapshot_features_are_flagged():
    assert find_forbidden_feature_columns(
        ["rolling_3_avg", "business_review_count", "business_stars"]
    ) == ["business_review_count", "business_stars"]


def test_select_safe_feature_columns_excludes_targets_predictions_and_future_features():
    columns = [
        "monthly_review_count",
        "rolling_3m_avg",
        "target_next_month_reviews",
        "attention_pulse",
        "pulse_relative_lift",
        "future_reviews_lead1",
        "target_month_str",
        "feature_month_str",
    ]

    assert select_safe_feature_columns(columns) == ["monthly_review_count", "rolling_3m_avg"]


def test_modeling_frame_flags_temporal_leakage():
    frame = pd.DataFrame(
        {
            "feature_month_str": ["2019-01", "2019-02"],
            "target_month_str": ["2019-02", "2019-02"],
            "rolling_3_avg": [1.0, 2.0],
        }
    )

    findings = check_modeling_frame(frame)

    assert any("strictly earlier" in finding.message for finding in findings)


def test_modeling_frame_flags_forbidden_columns_and_future_targets():
    frame = pd.DataFrame(
        {
            "feature_month_str": ["2019-12"],
            "target_month_str": ["2020-01"],
            "business_stars": [4.5],
        }
    )

    findings = check_modeling_frame(frame)

    assert any("Forbidden snapshot" in finding.message for finding in findings)
    assert any("exceed" in finding.message for finding in findings)


def test_modeling_frame_passes_clean_rows():
    frame = pd.DataFrame(
        {
            "feature_month_str": ["2018-12"],
            "target_month_str": ["2019-01"],
            "rolling_3_avg": [2.0],
        }
    )

    assert check_modeling_frame(frame) == []


def test_sna_summary_flags_future_review_window():
    findings = check_sna_summary({"review_window_end": "2018-01"})

    assert any(
        "2018-01" in finding.message and "2017-12" in finding.message for finding in findings
    )
