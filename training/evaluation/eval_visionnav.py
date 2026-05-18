"""
VisionNav Internal Benchmark — 100 handcrafted tasks.
All tasks run in isolated sandboxed environments.

Usage:
    python training/evaluation/eval_visionnav.py --model checkpoints/stage3_best
"""
from __future__ import annotations
import argparse

TASKS = [
    {"id": "br_001", "task": "Open Chrome and search for AI news",  "platform": "desktop"},
    {"id": "br_002", "task": "Navigate to github.com",              "platform": "desktop"},
    {"id": "em_001", "task": "Open Gmail and find unread emails",   "platform": "desktop"},
    {"id": "sy_001", "task": "Open system settings",                "platform": "desktop"},
    {"id": "fm_001", "task": "Fill a simple contact form",          "platform": "desktop"},
    {"id": "an_001", "task": "Open Settings on Android",            "platform": "android"},
    # Add more tasks as the project grows
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True)
    parser.add_argument("--platform", default="all")
    args  = parser.parse_args()
    tasks = [t for t in TASKS
             if args.platform == "all" or t["platform"] == args.platform]
    print(f"\nVisionNav Benchmark — {len(tasks)} tasks | model: {args.model}")
    print("(stub) — implement after Phase 6 agent loop is complete")


if __name__ == "__main__":
    main()
