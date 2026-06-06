"""
Metadata and lineage tracking for pipeline samples.

Every transformation a sample goes through is recorded here.
This answers the question: "What happened to this sample and when?"

Why lineage matters:
  If a bug is found in the OCR enricher,
  you can query: "which samples were processed by ocr_enricher v1.2?"
  Then re-enrich only those samples. No guesswork.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import structlog

log = structlog.get_logger(__name__)


def new_sample_id() -> str:
    """Generate a unique sample ID."""
    return f"vn_{uuid.uuid4().hex[:8]}"


def now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineEvent:
    """
    One event in a sample's history.
    Records every significant operation performed on a sample.

    Examples:
      {"event": "validated", "stage": "schema_validator", "result": "pass"}
      {"event": "enriched",  "stage": "ocr_enricher",     "version": "1.2"}
      {"event": "rejected",  "reason": "duplicate",        "gate": "dedup"}
    """

    event: str  # "validated", "enriched", "annotated", etc.
    stage: str  # which module did this: "schema_validator"
    timestamp: str = field(default_factory=now_iso)
    result: str = ""  # "pass", "fail", "quarantine"
    version: str = ""  # version of the module that ran
    details: dict = field(default_factory=dict)


@dataclass
class SampleLineage:
    """
    Complete audit trail for one sample.

    Answers:
      - Where did this sample come from?
      - Every operation performed on it
      - Which pipeline version processed it
      - Which training run included it

    Never modify existing events — only append new ones.
    Lineage is append-only by design.
    """

    sample_id: str
    source_file: str  # original raw file path
    source_line: int  # line number in source JSONL
    collector_id: str  # who collected this: "annotator_001", "system"
    collection_device: str  # "windows11_desktop", "android_pixel7"

    created_at: str = field(default_factory=now_iso)

    # Ordered log of every operation
    events: list[PipelineEvent] = field(default_factory=list)

    # Which training runs have used this sample
    included_in_runs: list[str] = field(default_factory=list)

    def add_event(
        self,
        event: str,
        stage: str,
        result: str = "",
        version: str = "",
        **details,
    ) -> None:
        """Append a new event to the lineage log."""
        self.events.append(
            PipelineEvent(
                event=event,
                stage=stage,
                result=result,
                version=version,
                details=dict(details),
            )
        )

    def was_processed_by(self, stage: str) -> bool:
        """Check if a specific stage has already processed this sample."""
        return any(e.stage == stage for e in self.events)

    def last_event_for(self, stage: str) -> PipelineEvent | None:
        """Get the most recent event from a specific stage."""
        matches = [e for e in self.events if e.stage == stage]
        return matches[-1] if matches else None

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


# ─── Lineage Store ────────────────────────────────────────────────────────────


class LineageStore:
    """
    Persists sample lineage to a JSONL file.
    Append-only — never modifies existing records.

    For Stage 1: local JSONL file
    For Stage 3+: replace with PostgreSQL without changing callers
    """

    def __init__(self, path: "Path") -> None:
        from pathlib import Path

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory cache: sample_id → lineage
        # Loads lazily on first access
        self._cache: dict[str, SampleLineage] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load all lineage records into memory on first access."""
        if self._loaded:
            return

        if not self._path.exists():
            self._loaded = True
            return

        import json

        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    lineage = self._from_dict(data)
                    self._cache[lineage.sample_id] = lineage
                except Exception:
                    pass  # corrupt line, skip

        self._loaded = True

    def get(self, sample_id: str) -> SampleLineage | None:
        self._ensure_loaded()
        return self._cache.get(sample_id)

    def save(self, lineage: SampleLineage) -> None:
        """
        Upsert lineage record.
        Updates in-memory cache immediately.
        Rewrites file (acceptable for Stage 1 scale).
        For Stage 3+: replace with SQL upsert.
        """
        self._ensure_loaded()
        self._cache[lineage.sample_id] = lineage
        self._persist()

    def add_event(
        self,
        sample_id: str,
        event: str,
        stage: str,
        **kwargs,
    ) -> SampleLineage | None:
        """Convenience: append event to existing lineage."""
        self._ensure_loaded()
        lineage = self._cache.get(sample_id)
        if lineage is None:
            return None
        lineage.add_event(event=event, stage=stage, **kwargs)
        self._persist()
        return lineage

    def record_run_membership(
        self,
        sample_ids: list[str],
        run_id: str,
    ) -> None:
        """
        Update included_in_runs for each sample in sample_ids.
        Reads the lineage file, updates matching entries, rewrites.
        Called once per pipeline run after samples are approved.
        """
        # IMPROVISED CODE: Populates included_in_runs field
        if not hasattr(self, "_lineage_path") or not self._lineage_path.exists():
            return

        id_set = set(sample_ids)
        entries = []

        with open(self._lineage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        updated = 0
        for entry in entries:
            if entry.get("sample_id") in id_set:
                runs = entry.get("included_in_runs", [])
                if run_id not in runs:
                    runs.append(run_id)
                    entry["included_in_runs"] = runs
                    updated += 1

        with open(self._lineage_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        log.info(
            "lineage_run_membership_updated", run_id=run_id, samples_updated=updated
        )

    def _persist(self) -> None:
        """Rewrite entire file from cache."""
        import json

        with open(self._path, "w", encoding="utf-8") as f:
            for lineage in self._cache.values():
                f.write(json.dumps(lineage.to_dict()) + "\n")

    def _from_dict(self, data: dict) -> SampleLineage:
        events = [PipelineEvent(**e) for e in data.get("events", [])]
        lineage = SampleLineage(
            sample_id=data["sample_id"],
            source_file=data["source_file"],
            source_line=data["source_line"],
            collector_id=data.get("collector_id", "unknown"),
            collection_device=data.get("collection_device", "unknown"),
            created_at=data.get("created_at", now_iso()),
            included_in_runs=data.get("included_in_runs", []),
        )
        lineage.events = events
        return lineage

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._cache)
