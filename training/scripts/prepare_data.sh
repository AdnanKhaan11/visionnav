#!/usr/bin/env bash
# Run full data preparation pipeline
set -euo pipefail
echo "========================================"
echo "VisionNav — Data Preparation Pipeline"
echo "========================================"
uv run python -m data_pipeline.pipeline --stage all
echo ""
echo "Data ready. Next: make train-stage1"
