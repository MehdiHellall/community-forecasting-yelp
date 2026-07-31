"""Input/output helpers used by notebooks, CLIs, and tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd


class JsonlFormatError(ValueError):
    """Raised when a JSONL file contains invalid or unexpected records."""


def iter_jsonl(path: str | Path, *, skip_blank: bool = True) -> Iterator[dict[str, Any]]:
    """Yield object records from a JSON Lines file.

    The Yelp Open Dataset ships as JSONL, so this helper keeps parsing behavior
    consistent across notebooks, CLIs, and tests.
    """
    jsonl_path = Path(path)
    with jsonl_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if skip_blank and not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{jsonl_path}:{line_number} is not valid JSON: {exc.msg}"
                raise JsonlFormatError(msg) from exc
            if not isinstance(record, dict):
                msg = f"{jsonl_path}:{line_number} must contain a JSON object"
                raise JsonlFormatError(msg)
            yield record


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write object records as UTF-8 JSON Lines."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def read_csv_with_columns(path: str | Path, required_columns: Iterable[str]) -> pd.DataFrame:
    """Read a CSV and fail fast when required columns are missing."""
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    return frame
