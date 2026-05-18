"""Pre-compute OCR for all training screenshots and cache results as JSON."""
from __future__ import annotations
from pathlib import Path

OCR_CACHE = Path("data/processed/ocr_cache")


def run_ocr_enrichment() -> None:
    OCR_CACHE.mkdir(parents=True, exist_ok=True)
    print("  Running PaddleOCR on processed screenshots...")
    # TODO Phase 2: batch PaddleOCR over all images
    # TODO Phase 2: save list[TextRegion] as JSON per image
    print("  (stub) OCR enrichment complete")
