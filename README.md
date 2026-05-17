# Community Forecasting with Yelp Data

This project studies **local community attention dynamics** on Yelp. It combines time-series review history, business metadata, social-network exposure, and lightweight NLP features to forecast which New Orleans businesses will receive future review activity or short-term attention pulses.

Research question:

> Can time-series history, social-network exposure, and review-language signals help forecast short-term shifts in community attention toward local Yelp businesses?

## Current Status

The notebook pipeline has been run end to end for **New Orleans, Louisiana**.

- Extracted **6,215 businesses**, **635,521 reviews**, and **245,421 reviewing users**.
- Built a weighted active-reviewer friendship graph with **26,598 nodes** and **116,558 edges**.
- Created a modeling table with **876 active businesses** and **68,249 business-month rows** after filtering to the 2015-2021 modeling window.
- Added two prediction tasks:
  - next-month review-count regression;
  - attention-pulse classification for unusually high next-month activity.
- Compared simple baselines, Random Forests, HistGradientBoosting, Poisson/logistic models, selected-feature variants, and pulse probability diagnostics.
- Added validation-tuned attention-pulse thresholds and top-k pulse retrieval metrics.
- Added capped TF-IDF topic indicators, pulse-predecessor analysis, and report-ready attention-pulse case studies.


## Repository Structure

```text
data/
  raw/yelp/       original Yelp JSON files, ignored by Git
  interim/        city-level extracted data, ignored by Git
  processed/      modeling-ready data, ignored by Git
notebooks/        reproducible academic workflow
outputs/          report metrics, predictions, figures, and interpretation tables
```

## Data

Place the Yelp Open Dataset files in `data/raw/yelp/`:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_user.json
yelp_academic_dataset_review.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_tip.json
```

Raw, interim, and processed data are not committed because of size. Report-ready outputs under `outputs/` are committed so the figures and summary tables can be reviewed without rerunning the full workflow.

## Notebook Workflow

```text
01_dataset_overview_and_city_selection.ipynb
02_new_orleans_data_preparation.ipynb
03_exploratory_analysis.ipynb
04_social_network_analysis.ipynb
05_time_series_feature_engineering.ipynb
06_forecasting_models.ipynb
07_results_interpretation.ipynb
```

The notebooks move from scope selection and extraction to EDA, SNA design, feature engineering, modeling, and final interpretation.

## Generated Outputs

The current local run creates:

```text
data/interim/new_orleans/
data/processed/new_orleans/
outputs/city_business_counts.csv
outputs/forecasting_metrics.csv
outputs/attention_pulse_metrics.csv
outputs/attention_pulse_topk_metrics.csv
outputs/attention_pulse_calibration.csv
outputs/model_comparison_summary.csv
outputs/nlp_tfidf_terms.csv
outputs/pulse_predecessor_analysis.csv
outputs/attention_pulse_case_studies.csv
outputs/figures/
```

The figure folders are organized by pipeline stage:

```text
outputs/figures/eda/
outputs/figures/sna/
outputs/figures/feature_engineering/
outputs/figures/models/
```

Full prediction files are regenerated locally by notebook `06`:

```text
outputs/forecasting_predictions.csv
outputs/attention_pulse_predictions.csv
```

They are intentionally ignored by Git because they are large and can be recreated from the notebooks.

## Reproducibility

Run the notebooks in this exact order:

```text
01_dataset_overview_and_city_selection.ipynb
02_new_orleans_data_preparation.ipynb
03_exploratory_analysis.ipynb
04_social_network_analysis.ipynb
05_time_series_feature_engineering.ipynb
06_forecasting_models.ipynb
07_results_interpretation.ipynb
```

Expected tracked outputs after a full run:

- metrics and summary tables in `outputs/*.csv`;
- figures in `outputs/figures/eda/`, `outputs/figures/sna/`, `outputs/figures/feature_engineering/`, and `outputs/figures/models/`;
- modeling-ready data in `data/processed/new_orleans/`, regenerated locally but ignored by Git.

The full prediction CSVs are local regeneration artifacts and are not committed.

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```
