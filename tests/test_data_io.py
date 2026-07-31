from __future__ import annotations

import re

import pandas as pd
import pytest

from community_forecasting.data_io import (
    JsonlFormatError,
    iter_jsonl,
    read_csv_with_columns,
    write_jsonl,
)


def test_iter_jsonl_reads_objects(tmp_path):
    path = tmp_path / "sample.jsonl"
    write_jsonl([{"id": 1}, {"id": 2}], path)

    assert list(iter_jsonl(path)) == [{"id": 1}, {"id": 2}]


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")

    assert [record["id"] for record in iter_jsonl(path)] == [1, 2]


def test_iter_jsonl_rejects_malformed_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(JsonlFormatError, match=re.escape("bad.jsonl:2")):
        list(iter_jsonl(path))


def test_iter_jsonl_rejects_non_object_records(tmp_path):
    path = tmp_path / "array.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(JsonlFormatError, match="must contain a JSON object"):
        list(iter_jsonl(path))


def test_read_csv_with_columns_fails_fast_on_missing_columns(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"present": [1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        read_csv_with_columns(path, ["present", "missing"])
