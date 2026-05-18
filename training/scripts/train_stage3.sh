#!/usr/bin/env bash
# Stage 3: Multi-Step Planning fine-tune
# Gate: ScreenSpot >= 79.6%, Mind2Web Step SR >= 44%
set -euo pipefail
echo "Starting Stage 3 — Multi-Step Planning"
uv run llamafactory-cli train configs/training/sft_3b_stage3.yaml
echo "Stage 3 complete. Run full eval: make eval"
