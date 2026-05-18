#!/usr/bin/env bash
# Run all linting and type checking
set -euo pipefail
echo "Running ruff..."
uv run ruff check src/ tests/ data_pipeline/
uv run ruff format --check src/ tests/
echo "Running mypy..."
uv run mypy src/visionnav --ignore-missing-imports
echo "All checks passed!"
