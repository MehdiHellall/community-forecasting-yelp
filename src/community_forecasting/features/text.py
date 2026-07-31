"""Leakage-aware text feature helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class FrozenTfidf:
    """A TF-IDF vectorizer fit only on training text."""

    vectorizer: TfidfVectorizer
    feature_columns: tuple[str, ...]


def safe_tfidf_column_name(term: str, *, prefix: str = "tfidf_recent") -> str:
    """Convert a TF-IDF term into a stable dataframe column name."""
    token = re.sub(r"[^a-zA-Z0-9]+", "_", term.strip().lower()).strip("_")
    token = token or "term"
    return f"{prefix}_{token}"


def fit_tfidf_on_train(
    train_text: pd.Series,
    *,
    max_features: int = 50,
    min_df: int = 1,
    max_df: float = 0.80,
    vectorizer_kwargs: dict[str, Any] | None = None,
) -> FrozenTfidf:
    """Fit a TF-IDF vocabulary on train-window text only."""
    kwargs = {
        "max_features": max_features,
        "min_df": min_df,
        "max_df": max_df,
        "stop_words": "english",
        "ngram_range": (1, 2),
    }
    kwargs.update(vectorizer_kwargs or {})
    vectorizer = TfidfVectorizer(**kwargs)
    vectorizer.fit(train_text.fillna("").astype(str))
    feature_columns = tuple(_dedupe_columns(vectorizer.get_feature_names_out()))
    return FrozenTfidf(vectorizer=vectorizer, feature_columns=feature_columns)


def transform_tfidf(frozen: FrozenTfidf, text: pd.Series) -> pd.DataFrame:
    """Transform text with a frozen train-window TF-IDF vocabulary."""
    matrix = frozen.vectorizer.transform(text.fillna("").astype(str))
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=frozen.feature_columns)
    dense = frame.sparse.to_dense()
    dense.index = text.index
    return dense


def _dedupe_columns(terms) -> list[str]:
    used: set[str] = set()
    columns: list[str] = []
    for term in terms:
        base_name = safe_tfidf_column_name(str(term))
        column_name = base_name
        suffix = 2
        while column_name in used:
            column_name = f"{base_name}_{suffix}"
            suffix += 1
        used.add(column_name)
        columns.append(column_name)
    return columns
