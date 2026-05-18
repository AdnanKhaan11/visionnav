#!/usr/bin/env bash
# Stage 1: GUI Grounding fine-tune
# Gate: ScreenSpot Acc@0.5 >= 75% before moving to Stage 2
set -euo pipefail
echo "Starting Stage 1 — GUI Grounding"
uv run llamafactory-cli train configs/training/sft_3b.yaml
echo "Stage 1 complete. Run: python training/evaluation/eval_screenspot.py"
