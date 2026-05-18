"""Dataset cleaning — deduplication, resolution normalisation, corrupt removal."""
from __future__ import annotations
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def run_cleaning() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("  Cleaning: dedup → normalise resolution → filter corrupt")
    # TODO Phase 1: imagehash perceptual dedup (threshold hamming < 8)
    # TODO Phase 1: resize all to 1280x720 (desktop) / 1080x1920 (mobile)
    # TODO Phase 1: remove files < 50 KB or blank screens
    print("  (stub) Cleaning complete")
