# Model Card

## Intended Use

The project forecasts short-term Yelp attention for New Orleans businesses. It is a portfolio project, not a production decision system.

## Tasks

| Task | Target | Primary metric |
| --- | --- | --- |
| Review-count regression | next-month review count | WAPE |
| Attention-pulse classification | unusually high next-month activity | F1, PR-AUC, top-k precision |

## Evaluation Design

The evaluation is chronological:

```text
train:      2015-02 to 2017-12
validation: 2018-01 to 2018-12
test:       2019-01 to 2019-12
```

Validation is used for threshold selection. The 2019 test year is reserved for final reporting.

## Feature Families

- Historical activity: lagged counts, rolling averages, cumulative review history, seasonality.
- Business flags: category-derived restaurant, food, nightlife, and tourism indicators.
- Social exposure: reviewer-network features built from train-window social graph data.
- NLP: recent text length, lexicon indicators, VADER sentiment, and capped TF-IDF features.

Snapshot Yelp `business_review_count` and `business_stars` are intentionally excluded from model features because they can include future activity relative to a feature month.

## Current Tracked Results

| Metric | Value |
| --- | --- |
| Best count WAPE | 0.376 |
| Best count model | rolling 3-month baseline |
| Best pulse F1 | 0.287 |
| Best pulse PR-AUC | 0.237 |
| Best top-10% pulse precision | 0.293 |

## Limitations

- Yelp review activity is only a proxy for community attention.
- Offline events, local news, tourism shocks, and platform behavior are not fully observed.
- Attention pulses are rule-defined labels, not ground-truth business outcomes.
- The analysis is pre-COVID only and should not be assumed to generalize to pandemic or post-pandemic behavior.
- Full reproduction requires local access to the Yelp Open Dataset.

## Leakage Controls

- Social-network features use the training window only; see [sna_training_window.md](sna_training_window.md).
- Cohort eligibility is based on train-window activity.
- TF-IDF vocabulary is fit on train-window text, then applied to validation/test text.
- Snapshot rating/review-count fields are excluded from model features.
- `cf-yelp leakage-check` scans notebook feature definitions for forbidden snapshot fields.
