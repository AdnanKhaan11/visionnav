#!/usr/bin/env bash
# Stage 2: Action Prediction fine-tune
# Gate: Action type accuracy >= 85%
set -euo pipefail
echo "Starting Stage 2 — Action Prediction"
uv run llamafactory-cli train configs/training/sft_3b_stage2.yaml
echo "Stage 2 complete."
