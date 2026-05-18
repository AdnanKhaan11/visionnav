#!/usr/bin/env bash
# One-command dev environment setup
set -euo pipefail

echo "Setting up VisionNav dev environment..."

# Check uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Sync all dependencies
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Copy .env if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — add your tokens!"
fi

echo "Done! Run: make test-unit"
