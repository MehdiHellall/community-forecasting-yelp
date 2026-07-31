from __future__ import annotations

import pandas as pd

from community_forecasting.features.text import (
    fit_tfidf_on_train,
    safe_tfidf_column_name,
    transform_tfidf,
)


def test_safe_tfidf_column_name_is_stable():
    assert safe_tfidf_column_name("Great Food!") == "tfidf_recent_great_food"


def test_tfidf_vocab_is_fit_on_train_only():
    train_text = pd.Series(["gumbo po boy", "gumbo brunch", "po boy"])
    validation_text = pd.Series(["future-only oyster", "gumbo"])

    frozen = fit_tfidf_on_train(
        train_text,
        min_df=1,
        max_df=1.0,
        vectorizer_kwargs={"stop_words": None, "ngram_range": (1, 1)},
    )
    transformed = transform_tfidf(frozen, validation_text)

    assert "tfidf_recent_future_only" not in transformed.columns
    assert "tfidf_recent_oyster" not in transformed.columns
    assert "tfidf_recent_gumbo" in transformed.columns
    assert len(transformed) == 2


def test_transform_tfidf_preserves_input_index():
    train_text = pd.Series(["gumbo po boy", "gumbo brunch", "po boy"])
    validation_text = pd.Series(["gumbo", "po boy"], index=[10, 20])

    frozen = fit_tfidf_on_train(
        train_text,
        min_df=1,
        max_df=1.0,
        vectorizer_kwargs={"stop_words": None, "ngram_range": (1, 1)},
    )

    assert transform_tfidf(frozen, validation_text).index.tolist() == [10, 20]
