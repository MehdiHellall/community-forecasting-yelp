# SNA Training Window and Leakage Guard

## What changed

The social network analysis now builds reviewer activity, active-reviewer thresholds, edge weights, centrality scores, components, and community labels using only New Orleans reviews from **2015-02 through 2017-12**.

That date range matches the `target_month_str` training split used later in `notebooks/06_forecasting_models.ipynb`:

- train: `2015-02` to `2017-12`
- validation: `2018-01` to `2018-12`
- test: `2019-01` to `2019-12`

## Why this matters

Previously, the SNA notebook used the full local review history through 2021 to decide which users were active and to compute shared-business/category overlap. Those SNA outputs were then merged into the forecasting table. Because validation and test months occur after 2017, using post-2017 review activity would let future behavior influence features used for forecasting.

The train-only SNA graph prevents that future leakage. Reviewers who only appear after 2017 no longer receive learned centrality or community labels from future data; they fall back to the zero/default social feature values during feature engineering. This is intentional: weaker social coverage in validation and test is the honest cost of avoiding look-ahead information.

## Outputs affected

Rerunning `notebooks/04_social_network_analysis.ipynb` refreshes:

- `data/processed/new_orleans/user_network_features.csv`
- `data/processed/new_orleans/active_reviewer_threshold_sensitivity.csv`
- `data/processed/new_orleans/social_graph_summary.json`
- `outputs/figures/sna/*.png`

The graph summary records the SNA review window so future readers can audit the leakage guard directly.
