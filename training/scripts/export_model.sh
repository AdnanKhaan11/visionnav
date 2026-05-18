#!/usr/bin/env bash
# Export fine-tuned model: merge LoRA → quantize → package
set -euo pipefail
CHECKPOINT="${1:-checkpoints/stage3_planning}"
OUTPUT_DIR="models/visionnav-3b"
echo "Exporting from: $CHECKPOINT"
echo "Output to     : $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
# TODO Phase 8: llamafactory-cli export + AWQ/GGUF quantization
echo "(stub) Export pipeline — implement in Phase 8"
