"""
Pipeline metrics — tracks health across all pipeline stages.

Tracks:
  - Throughput (samples/hour per stage)
  - Rejection rate (per gate, per source)
  - Quality distribution (score buckets)
  - Language distribution
  - Stage routing distribution
  - Running totals since pipeline start
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class StageCounter:
    """Counters for one pipeline stage."""

    processed: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.processed, 1)

    @property
    def fail_rate(self) -> float:
        return self.failed / max(self.processed, 1)


@dataclass
class PipelineMetrics:
    """
    Tracks metrics across one pipeline run.

    One instance per pipeline.run() call.
    Serializable for storage/reporting.
    """

    run_id: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: str = ""

    # ── Stage counters ─────────────────────────────────────────────────────
    ingestion: StageCounter = field(default_factory=StageCounter)
    validation: StageCounter = field(default_factory=StageCounter)
    enrichment: StageCounter = field(default_factory=StageCounter)
    annotation: StageCounter = field(default_factory=StageCounter)
    curation: StageCounter = field(default_factory=StageCounter)
    export: StageCounter = field(default_factory=StageCounter)

    # ── Rejection reasons ─────────────────────────────────────────────────
    # gate_name → count
    rejection_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── Quality distribution ───────────────────────────────────────────────
    # bucket → count: "0.9-1.0" → 150, "0.8-0.9" → 80, etc.
    quality_buckets: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── Language distribution ──────────────────────────────────────────────
    language_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── Stage routing distribution ─────────────────────────────────────────
    stage_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ── Export results ─────────────────────────────────────────────────────
    exported_per_stage: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_quality(self, score: float) -> None:
        """Increment the correct quality bucket for a score."""
        if score >= 0.9:
            bucket = "0.90-1.00"
        elif score >= 0.8:
            bucket = "0.80-0.90"
        elif score >= 0.7:
            bucket = "0.70-0.80"
        elif score >= 0.6:
            bucket = "0.60-0.70"
        else:
            bucket = "0.00-0.60"
        self.quality_buckets[bucket] = self.quality_buckets.get(bucket, 0) + 1

    def record_rejection(self, reason: str) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def record_language(self, language: str) -> None:
        self.language_counts[language] = self.language_counts.get(language, 0) + 1

    def record_stage_route(self, stage: str) -> None:
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        """Serialize to plain dict for storage."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ingestion": {
                "processed": self.ingestion.processed,
                "passed": self.ingestion.passed,
                "failed": self.ingestion.failed,
            },
            "validation": {
                "processed": self.validation.processed,
                "passed": self.validation.passed,
                "failed": self.validation.failed,
            },
            "enrichment": {
                "processed": self.enrichment.processed,
                "passed": self.enrichment.passed,
                "failed": self.enrichment.failed,
            },
            "annotation": {
                "processed": self.annotation.processed,
                "passed": self.annotation.passed,
                "failed": self.annotation.failed,
            },
            "curation": {
                "processed": self.curation.processed,
                "passed": self.curation.passed,
                "failed": self.curation.failed,
            },
            "rejection_reasons": dict(self.rejection_reasons),
            "quality_buckets": dict(self.quality_buckets),
            "language_counts": dict(self.language_counts),
            "stage_counts": dict(self.stage_counts),
            "exported_per_stage": dict(self.exported_per_stage),
        }
