"""
Test: Batch 3 — annotation, curation, registry end-to-end.
Extends the Batch 2 test with full pipeline including quality scoring.

Run: python test_pipeline_batch3.py
"""

import sys, json
from pathlib import Path

sys.path.insert(0, "src")

PIPELINE_DIR = Path("data/pipeline_test")
DB_PATH = PIPELINE_DIR / "registry.db"
LINEAGE_FILE = PIPELINE_DIR / "lineage.jsonl"
DEDUP_INDEX = PIPELINE_DIR / ".dedup_index.json"

# ── Build components ──────────────────────────────────────────────────────────
from data_pipeline.core.metadata import LineageStore
from data_pipeline.core.schemas import (
    SampleStatus,
    SourceType,
    EnrichedContent,
    Annotation,
)
from data_pipeline.core.metadata import now_iso, new_sample_id
from data_pipeline.core.schemas import PipelineSample, RawContent, QualityMetrics
from data_pipeline.validation.gate import ValidationGate
from data_pipeline.validation import (
    schema_validator,
    coordinate_validator,
    image_validator,
)
from data_pipeline.validation.deduplicator import Deduplicator
from data_pipeline.enrichment.enricher import SampleEnricher
from data_pipeline.annotation.difficulty_scorer import score as score_difficulty
from data_pipeline.curation.quality_scorer import compute as compute_quality
from data_pipeline.curation.stage_router import route as route_stage
from data_pipeline.registry.registry import DatasetRegistry
from data_pipeline.registry.versioning import DatasetVersion

lineage = LineageStore(LINEAGE_FILE)
dedup = Deduplicator(index_path=DEDUP_INDEX)
gate = ValidationGate(lineage_store=lineage)
gate.register("schema", schema_validator.validate, is_fatal=True)
gate.register("coordinates", coordinate_validator.validate, is_fatal=True)
gate.register("image", image_validator.validate, is_fatal=False)
gate.register("dedup", dedup.validate, is_fatal=True)

enricher = SampleEnricher(lineage_store=lineage)
registry = DatasetRegistry(DB_PATH)

# ── Create a high-quality mock sample (bypasses image gate for testing) ───────
sample = PipelineSample(
    sample_id=new_sample_id(),
    session_id="test_session_001",
    step_index=3,
    source_type=SourceType.HUMAN_DEMO.value,
    task="Open Gmail and reply to the most recent unread email",
    status=SampleStatus.RAW.value,
    created_at=now_iso(),
    updated_at=now_iso(),
    raw=RawContent(
        screenshot_path="mock_path.png",
        action_type="click",
        coordinates=[0.5, 0.22],
        text=None,
        key=None,
        description="Click the most recent unread email",
        ocr_text_raw="Gmail Inbox Unread 3 messages John Smith Hello team meeting",
    ),
    enriched=EnrichedContent(
        ocr_regions_detailed=[
            {
                "text": "Gmail",
                "bbox": [0.0, 0.0, 0.1, 0.03],
                "confidence": 0.98,
                "script": "latin",
            },
            {
                "text": "Inbox",
                "bbox": [0.1, 0.0, 0.2, 0.03],
                "confidence": 0.97,
                "script": "latin",
            },
            {
                "text": "John Smith",
                "bbox": [0.1, 0.2, 0.5, 0.25],
                "confidence": 0.96,
                "script": "latin",
            },
            {
                "text": "Hello team",
                "bbox": [0.1, 0.25, 0.7, 0.3],
                "confidence": 0.94,
                "script": "latin",
            },
            {
                "text": "Reply",
                "bbox": [0.7, 0.9, 0.8, 0.95],
                "confidence": 0.99,
                "script": "latin",
            },
        ],
        image_hash_sha256="abc123def456",
        image_hash_phash="0f0f0f0f0f0f0f0f",
        detected_language="en",
        detected_platform="web",
        detected_app="gmail",
        n_ocr_regions=5,
        image_width=1920,
        image_height=1080,
        image_file_size_kb=245.0,
    ),
    annotation=Annotation(
        reasoning="I can see the Gmail inbox with 3 unread messages. John Smith's email appears at the top as the most recent. I need to click on it to open the email before I can reply.",
        intent="Open the most recent unread email to access the reply option",
        difficulty=0,  # will be set by scorer
        annotated_by="claude-haiku-4-5-20251001",
        annotation_method="auto_llm",
        annotation_confidence=0.85,
        reasoning_verified=True,
        verified_at=now_iso(),
    ),
    quality=QualityMetrics(
        schema_valid=True,
        image_valid=True,
        coordinates_valid=True,
        no_pii_detected=True,
        not_duplicate=True,
    ),
    tags=["email", "browser", "action:click"],
)

print("\n" + "═" * 60)
print("Dataset Factory — Batch 3 Integration Test")
print("═" * 60)

# ── Step 1: Difficulty scoring ────────────────────────────────────────────────
difficulty = score_difficulty(sample)
print(f"\n[1] Difficulty Score: {difficulty}/5")

# ── Step 2: Quality scoring ───────────────────────────────────────────────────
quality = compute_quality(sample, quality_threshold=0.65)
print(f"[2] Quality Score:    {quality:.4f}")
print(f"    Approved:         {sample.quality.approved_for_training}")
print(f"    Status:           {sample.status}")

# ── Step 3: Stage routing ─────────────────────────────────────────────────────
stage = route_stage(sample)
print(f"[3] Training Stage:   {stage or 'none (not routed)'}")

# ── Step 4: Registry ──────────────────────────────────────────────────────────
registry.register_sample(sample)
print(f"[4] Registered in DB: {DB_PATH}")

registry.create_version("1.0.0", notes="First test release")
print(f"    Created version:  1.0.0")

# ── Step 5: Stats ──────────────────────────────────────────────────────────────
stats = registry.stats()
print(f"\n[5] Registry Stats:")
print(f"    Total samples:    {stats['total_samples']}")
print(f"    Approved:         {stats['approved']} ({stats['approval_rate']}%)")
print(f"    Avg quality:      {stats['avg_quality_score']:.4f}")
print(f"    By language:      {stats['by_language']}")
print(f"    By stage:         {stats['by_stage']}")

# ── Step 6: Contamination check ───────────────────────────────────────────────
contaminated = registry.contamination_check([sample.sample_id, "fake_test_id"])
print(f"\n[6] Contamination check (training vs test):")
print(f"    Contaminated IDs: {contaminated}")
print(
    f"    {'⚠ WARNING: test set contamination!' if contaminated else '✓ No contamination'}"
)

# ── Step 7: Version list ──────────────────────────────────────────────────────
versions = registry.list_versions()
print(f"\n[7] Dataset Versions:")
for v in versions:
    print(
        f"    v{v['version']}: {v['approved_samples']} approved / {v['total_samples']} total"
    )

print("\n" + "═" * 60)
print("Batch 3 complete.")
print("═" * 60)
