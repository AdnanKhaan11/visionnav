"""
Test: Batch 1 + Batch 2 integration.
Ingest real recordings → validate → enrich → inspect results.

Run: python test_pipeline_batch2.py
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, "src")

# ── Setup paths ───────────────────────────────────────────────────────────────
RECORDINGS_DIR = Path("data/recordings")
PIPELINE_DIR = Path("data/pipeline_test")
LINEAGE_FILE = PIPELINE_DIR / "lineage.jsonl"
DEDUP_INDEX = PIPELINE_DIR / ".dedup_index.json"

PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

# ── Build pipeline components ─────────────────────────────────────────────────
from data_pipeline.core.metadata import LineageStore
from data_pipeline.core.schemas import SampleStatus
from data_pipeline.ingestion.sources.recorder_source import RecorderSource
from data_pipeline.validation.gate import ValidationGate
from data_pipeline.validation import (
    schema_validator,
    coordinate_validator,
    image_validator,
)
from data_pipeline.validation.deduplicator import Deduplicator
from data_pipeline.enrichment.enricher import SampleEnricher

# Lineage store
lineage = LineageStore(LINEAGE_FILE)

# Validation gate — register validators in priority order
dedup = Deduplicator(index_path=DEDUP_INDEX)
gate = ValidationGate(lineage_store=lineage)
gate.register("schema", schema_validator.validate, is_fatal=True)
gate.register("coordinates", coordinate_validator.validate, is_fatal=True)
gate.register("image", image_validator.validate, is_fatal=False)
gate.register("dedup", dedup.validate, is_fatal=True)

# Enricher
enricher = SampleEnricher(lineage_store=lineage)

# Source
source = RecorderSource(
    lineage_store=lineage,
    collector_id="test_run",
    device_name="windows_11",
)

# ── Run pipeline ──────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("Dataset Factory — Integration Test")
print("═" * 60)

# Check for recordings
if not RECORDINGS_DIR.exists():
    print(f"\nNo recordings directory found at: {RECORDINGS_DIR}")
    print("Creating a mock recording for testing...")

    # Create a mock recording
    import json, numpy as np
    from PIL import Image

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS = RECORDINGS_DIR / "screenshots" / "mock_session"
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)

    # Create a test screenshot
    img = Image.new("RGB", (1280, 720), color=(240, 240, 240))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "Gmail - Inbox (3 unread)", fill=(0, 0, 0))
    draw.text((100, 200), "Submit", fill=(255, 255, 255))
    screenshot_path = SCREENSHOTS / "step_000.png"
    img.save(screenshot_path)

    # Create mock JSONL
    mock_step = {
        "step_index": 0,
        "task": "Open Gmail and find unread emails",
        "session_id": "mock_session",
        "action": {
            "type": "click",
            "coordinates": [0.5, 0.22],
            "description": "Click the first unread email",
        },
        "screenshot_path": str(screenshot_path),
        "ocr_text": "Gmail Inbox 3 unread Submit",
        "timestamp": "2026-05-20T12:00:00Z",
    }
    jsonl_path = RECORDINGS_DIR / "session_mock_session.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(mock_step) + "\n")

    print(f"Mock recording created at: {jsonl_path}")


# ── Process ───────────────────────────────────────────────────────────────────
result = source.ingest_directory(RECORDINGS_DIR)
print(f"\nIngestion: {result.ingested} samples from {result.source_path}")

approved = []
rejected = []
quarantined = []

for sample in result.samples:
    gate_result = gate.run(sample)

    if gate_result.passed:
        enrich_report = enricher.enrich(sample)
        approved.append(sample)
        print(f"\n  ✓ {sample.sample_id}")
        print(f"    Task:     {sample.task[:60]}")
        print(f"    Action:   {sample.raw.action_type if sample.raw else '?'}")
        print(
            f"    Language: {sample.enriched.detected_language if sample.enriched else '?'}"
        )
        print(
            f"    OCR:      {sample.enriched.n_ocr_regions if sample.enriched else 0} regions"
        )
        print(
            f"    Hash:     {sample.enriched.image_hash_sha256[:16] if sample.enriched else ''}..."
        )
    elif sample.status == SampleStatus.QUARANTINED.value:
        quarantined.append(sample)
        print(f"\n  ⚠ {sample.sample_id} — QUARANTINED: {gate_result.reason[:60]}")
    else:
        rejected.append(sample)
        print(f"\n  ✗ {sample.sample_id} — REJECTED: {gate_result.reason[:60]}")


# ── Summary ───────────────────────────────────────────────────────────────────
total = len(result.samples)
print("\n" + "═" * 60)
print("Summary")
print("═" * 60)
print(f"Total ingested: {total}")
print(f"Approved:       {len(approved)}  ({len(approved)/max(total,1)*100:.0f}%)")
print(f"Quarantined:    {len(quarantined)}")
print(f"Rejected:       {len(rejected)}")
print(f"Lineage records: {lineage.count()}")
print("═" * 60)
