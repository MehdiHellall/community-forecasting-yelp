# Architecture

## System Shape

The repo now has two layers:

| Layer | Purpose |
| --- | --- |
| `notebooks/` | Human-readable research narrative and full local workflow. |
| `src/community_forecasting/` | Reusable, tested helpers and CLI checks for portfolio-grade engineering. |

## Package Modules

- `data_io.py`: JSONL streaming and CSV column validation.
- `splits.py`: chronological split definitions and dataframe partitioning.
- `metrics.py`: regression, classification, top-k, threshold, and calibration metrics.
- `features/text.py`: train-window TF-IDF fitting and transformation.
- `leakage.py`: snapshot-feature and temporal leakage guardrails.
- `validation.py`: committed output artifact checks.
- `cli.py`: `cf-yelp` commands.

## CLI Commands

```bash
cf-yelp summarize-results
cf-yelp validate-outputs
cf-yelp leakage-check
cf-yelp execute-notebooks --smoke
```

These commands are intentionally lightweight so they can run in CI without raw Yelp data.
