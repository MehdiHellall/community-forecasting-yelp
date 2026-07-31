"""Chronological split definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SplitLabel = Literal["train", "validation", "test", "out_of_scope"]


@dataclass(frozen=True)
class ChronologicalSplit:
    """Inclusive monthly train/validation/test split."""

    name: str
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def period_label(self, prefix: Literal["train", "validation", "test"]) -> str:
        return f"{getattr(self, f'{prefix}_start')} to {getattr(self, f'{prefix}_end')}"


DEFAULT_SPLIT = ChronologicalSplit(
    name="normal_pre_covid_test",
    train_start="2015-02",
    train_end="2017-12",
    validation_start="2018-01",
    validation_end="2018-12",
    test_start="2019-01",
    test_end="2019-12",
)


def label_month(month: str | pd.Period, split: ChronologicalSplit = DEFAULT_SPLIT) -> SplitLabel:
    """Label a target month according to the project's chronological split."""
    period = pd.Period(month, freq="M")
    if pd.Period(split.train_start, freq="M") <= period <= pd.Period(split.train_end, freq="M"):
        return "train"
    if (
        pd.Period(split.validation_start, freq="M")
        <= period
        <= pd.Period(split.validation_end, freq="M")
    ):
        return "validation"
    if pd.Period(split.test_start, freq="M") <= period <= pd.Period(split.test_end, freq="M"):
        return "test"
    return "out_of_scope"


def assign_split(
    frame: pd.DataFrame,
    *,
    month_column: str = "target_month_str",
    split: ChronologicalSplit = DEFAULT_SPLIT,
) -> pd.Series:
    """Return split labels for a dataframe month column."""
    if month_column not in frame.columns:
        raise ValueError(f"Missing month column: {month_column}")
    return frame[month_column].apply(lambda value: label_month(value, split))


def split_frame(
    frame: pd.DataFrame,
    *,
    month_column: str = "target_month_str",
    split: ChronologicalSplit = DEFAULT_SPLIT,
) -> dict[SplitLabel, pd.DataFrame]:
    """Partition a frame by chronological split label."""
    labeled = frame.assign(split_label=assign_split(frame, month_column=month_column, split=split))
    return {
        label: labeled[labeled["split_label"].eq(label)].drop(columns=["split_label"]).copy()
        for label in ["train", "validation", "test", "out_of_scope"]
    }
