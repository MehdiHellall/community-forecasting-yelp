# Community Forecasting with Yelp Data

This project explores how social network analysis and time series forecasting can be combined to forecast community activity on Yelp.

The academic objective is to build an interpretable forecasting workflow using:

- historical review activity,
- user and business metadata,
- social connections between users,
- and network-derived features.

The longer-term objective is to keep the work structured enough that it can later evolve into a production-style forecasting system. For the academic phase, the main analysis is kept in notebooks so that data preparation, reasoning, intermediate results, and interpretation are visible.

## Project Structure

```text
data/
  raw/
    yelp/          original Yelp JSON files, never edited
  interim/         filtered intermediate files
  processed/       modeling-ready datasets
  external/        any extra reference data
models/            trained model artifacts, ignored by Git
notebooks/         academic analysis and reproducible workflow
outputs/           generated metrics, plots, city counts, and predictions
reports/           report material and final write-up assets
```

A `src/` folder may be added later when the project moves from academic exploration toward a production system.

## Raw Data

Place the Yelp Open Dataset files in:

```text
data/raw/yelp/
```

Expected files:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_user.json
yelp_academic_dataset_review.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_tip.json
```

The raw files are ignored by Git because they are large and should not be committed.

## Notebook Workflow

The academic notebooks are:

```text
notebooks/01_dataset_overview_and_city_selection.ipynb
notebooks/02_new_orleans_data_preparation.ipynb
notebooks/03_exploratory_analysis.ipynb
notebooks/04_social_network_analysis.ipynb
notebooks/05_time_series_feature_engineering.ipynb
notebooks/06_forecasting_models.ipynb
notebooks/07_results_interpretation.ipynb
```

### 01 - Dataset Overview and City Selection

This notebook inspects the business table, counts businesses by city/state, and motivates the decision to focus on one city.

It creates:

```text
outputs/city_business_counts.csv
```

### 02 - New Orleans Data Preparation

This notebook extracts a New Orleans subset from the raw Yelp files by streaming through the business, review, and user JSONL files.

It creates:

```text
data/interim/new_orleans/businesses.jsonl
data/interim/new_orleans/reviews.jsonl
data/interim/new_orleans/users.jsonl
data/interim/new_orleans/summary.json
```

### 03 - Exploratory Analysis

This notebook begins exploration of the New Orleans subset, including review volume over time, business category distribution, review concentration by business, and user activity concentration.

### 04 - Social Network Analysis

This notebook builds an active-reviewer Yelp friendship graph and creates user-level SNA features such as degree, PageRank, component size, active reviewer flag, and community label where feasible.

### 05 - Time Series Feature Engineering

This notebook creates the one-month-ahead forecasting table. The main cohort uses businesses with at least 100 reviews and at least 36 active review months, excludes January 2022, and models targets from 2015-02 through 2021-12.

### 06 - Forecasting Models

This notebook compares naive baselines, rolling-average baselines, historical-feature ML models, historical-plus-business models, and historical-plus-business-plus-SNA models using chronological train/test splits.

### 07 - Results Interpretation

This notebook consolidates the forecasting metrics and interprets whether SNA features improved prediction beyond historical review patterns and business metadata.


## Main Research Objective

The main objective is to evaluate whether Yelp social-network features improve **one-month-ahead forecasting of monthly review activity** for active New Orleans businesses.

Research question:

> Can social interaction patterns between Yelp users help predict future business review activity beyond historical review trends alone?

The final modeling cohort contains **1,151 businesses** and **95,533 business-month rows** after applying the activity threshold and excluding the partial January 2022 month.

## Selected Scope

The project focuses on **New Orleans, Louisiana**.

The normalized city extraction selected **6,215 New Orleans businesses** from the Yelp Open Dataset. This is a practical academic scope: large enough to support meaningful social network and time series analysis, but smaller than the largest cities in the dataset.

The first completed extraction produced:

```text
Businesses:              6,215
Reviews:                 635,521
Unique reviewing users:  245,421
Matched user profiles:   245,419
Review date range:       2005-03-14 to 2022-01-19
```

## City Normalization Note

The initial city-count inspection and the final city extraction may produce slightly different counts. The inspection table is used for orientation, while the extraction step applies cleaner matching:

- city names are stripped of leading/trailing whitespace,
- city matching is case-insensitive,
- state abbreviations are stripped and uppercased.

For New Orleans, the first inspection showed **6,208** businesses, while the normalized extraction selected **6,215** businesses. The normalized count is used for the project because it better handles small formatting inconsistencies in the raw data.
