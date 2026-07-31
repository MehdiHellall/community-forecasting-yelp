from __future__ import annotations

import pandas as pd
import pytest

from community_forecasting.splits import DEFAULT_SPLIT, assign_split, label_month, split_frame


def test_label_month_uses_inclusive_split_boundaries():
    assert label_month("2015-02") == "train"
    assert label_month("2017-12") == "train"
    assert label_month("2018-01") == "validation"
    assert label_month("2018-12") == "validation"
    assert label_month("2019-01") == "test"
    assert label_month("2019-12") == "test"
    assert label_month("2020-01") == "out_of_scope"


def test_assign_split_labels_dataframe_rows():
    frame = pd.DataFrame({"target_month_str": ["2017-12", "2018-07", "2019-03", "2014-01"]})

    assert assign_split(frame).tolist() == ["train", "validation", "test", "out_of_scope"]


def test_split_frame_returns_all_partitions():
    frame = pd.DataFrame(
        {
            "target_month_str": ["2017-12", "2018-07", "2019-03"],
            "value": [1, 2, 3],
        }
    )

    partitions = split_frame(frame)

    assert partitions["train"]["value"].tolist() == [1]
    assert partitions["validation"]["value"].tolist() == [2]
    assert partitions["test"]["value"].tolist() == [3]
    assert partitions["out_of_scope"].empty
    assert DEFAULT_SPLIT.period_label("train") == "2015-02 to 2017-12"


def test_assign_split_requires_month_column():
    with pytest.raises(ValueError, match="Missing month column"):
        assign_split(pd.DataFrame({"month": ["2019-01"]}))
