"""
Mind2Web Benchmark Evaluation
================================
Metrics: Element Acc, Operation F1, Step Success Rate.

Usage:
    python training/evaluation/eval_mind2web.py --model checkpoints/stage3_best
"""
from __future__ import annotations
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="cross_task",
                        choices=["cross_task", "cross_website", "cross_domain"])
    args = parser.parse_args()
    print(f"\nMind2Web Evaluation — {args.split}")
    print(f"Model: {args.model}")
    print("(stub) — implement after Stage 3 training")


if __name__ == "__main__":
    main()
