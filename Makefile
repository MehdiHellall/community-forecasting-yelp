.PHONY: install lint format-check test validate smoke

install:
	uv sync --extra dev

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

test:
	uv run pytest

validate:
	uv run cf-yelp validate-outputs
	uv run cf-yelp leakage-check

smoke:
	uv run cf-yelp execute-notebooks --smoke
