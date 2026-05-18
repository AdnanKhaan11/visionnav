"""
ScreenSpot Benchmark Evaluation
=================================
Gate: must pass threshold before promoting model to next training stage.

Usage:
    python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best
    python training/evaluation/eval_screenspot.py --model checkpoints/stage1_best --threshold 75.0
"""
from __future__ import annotations
import argparse

THRESHOLDS = {
    "stage1": 75.0,   # after grounding training
    "stage2": 79.6,   # TongUI-3B baseline
    "stage3": 83.6,   # TongUI-3B-1M level
    "7b":     86.0,   # after 7B fine-tune
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--split",     default="test", choices=["val", "test"])
    args = parser.parse_args()

    print(f"\nScreenSpot Evaluation")
    print(f"Model     : {args.model}")
    print(f"Threshold : {args.threshold}%")
    print(f"{'='*50}")
    # TODO Phase 3: implement evaluation loop
    # Reference: TongUI-agent/tongui/eval/run_screenspot.py
    print("(stub) — implement after Stage 1 training is complete")


if __name__ == "__main__":
    main()
