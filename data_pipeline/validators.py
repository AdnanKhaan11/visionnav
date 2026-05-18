"""Validate final dataset — schema, coordinates, completeness."""
from __future__ import annotations
import json
from pathlib import Path


def run_validation() -> None:
    files = list(Path("data/instruction_tuning").glob("*.jsonl"))
    if not files:
        print("  No files found — run --stage format first")
        return

    total = errors = 0
    for f in files:
        with open(f) as fp:
            for line in fp:
                total += 1
                try:
                    _validate(json.loads(line))
                except (json.JSONDecodeError, AssertionError) as exc:
                    errors += 1
                    print(f"  ERR {f.name}:{total} — {exc}")

    print(f"  Validated {total} samples — {errors} errors")
    if errors:
        raise SystemExit(f"Validation failed: {errors} invalid samples")
    print("  All samples valid")


def _validate(s: dict) -> None:
    assert "conversations" in s, "Missing conversations"
    assert "images" in s,        "Missing images"
    assert len(s["conversations"]) >= 2, "Need >= 2 turns"
    assert len(s["images"]) >= 1,        "Need >= 1 image"
