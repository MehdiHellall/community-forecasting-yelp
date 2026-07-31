# Reproducibility

## Environment

Use Python 3.12.

Recommended setup:

```bash
uv sync --extra dev
```

Full notebook setup:

```bash
uv sync --extra dev --extra notebooks
```

Pip fallback:

```bash
python -m pip install -e ".[dev]"
```

The local development environment used for the rework already had the heavy notebook dependencies installed, but not `uv`, `pytest`, or `ruff`.

## CI-Safe Verification

These commands do not require raw Yelp data:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run cf-yelp validate-outputs
uv run cf-yelp leakage-check
uv run cf-yelp execute-notebooks --smoke
```

`execute-notebooks --smoke` parses notebooks and fails on saved error outputs. It does not execute the full data pipeline.

## Full Local Reproduction

1. Download the Yelp Open Dataset according to Yelp's terms.
2. Place the JSON files under `data/raw/yelp/`.
3. Run notebooks `01` through `07` in order.
4. Re-run the CI-safe verification commands above.

The full pipeline is intentionally notebook-driven for v1 because the portfolio goal is clean engineering around an existing analysis, not a full orchestration platform.

## Expected Committed Artifacts

After a full run, the repo should have:

- summary metrics in `outputs/*.csv`;
- report figures in `outputs/figures/`;
- raw/interim/processed data still ignored by Git;
- full prediction files still ignored by Git.

## Future Extensions

The pragmatic v1 intentionally skips MLflow, DVC, Docker, Streamlit, and FastAPI. Good stretch additions would be:

- MLflow for experiment tracking;
- DVC for large artifact lineage;
- Docker for fully pinned reproduction;
- Streamlit for a visual demo;
- FastAPI for a production-style scoring service.
