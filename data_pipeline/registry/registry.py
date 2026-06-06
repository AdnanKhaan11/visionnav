"""
DatasetRegistry — single source of truth for all dataset versions.

Stage 1: SQLite (local, zero setup, handles millions of samples)
Stage 3+: Replace with PostgreSQL by changing _get_connection() only.
          All queries stay identical — SQLite and PostgreSQL
          both speak standard SQL.

The registry tracks:
  - Every sample that has been processed
  - Which dataset version each sample belongs to
  - Sample quality and stage assignment
  - Which training runs have used each sample
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import structlog

from data_pipeline.core.exceptions import RegistryError
from data_pipeline.core.schemas import PipelineSample, SampleStatus
from data_pipeline.registry.versioning import DatasetVersion

log = structlog.get_logger(__name__)


# ─── Schema ───────────────────────────────────────────────────────────────────

_CREATE_SAMPLES_TABLE = """
CREATE TABLE IF NOT EXISTS samples (
    sample_id           TEXT PRIMARY KEY,
    session_id          TEXT NOT NULL,
    step_index          INTEGER NOT NULL,
    source_type         TEXT NOT NULL,
    task                TEXT NOT NULL,
    task_category       TEXT DEFAULT '',
    status              TEXT NOT NULL,
    quality_score       REAL DEFAULT 0.0,
    approved            INTEGER DEFAULT 0,
    training_stage      TEXT DEFAULT '',
    detected_language   TEXT DEFAULT 'unknown',
    action_type         TEXT DEFAULT '',
    schema_version      TEXT DEFAULT '1.0.0',
    dataset_version     TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    tags                TEXT DEFAULT '[]',   -- JSON array
    rejection_reason    TEXT DEFAULT ''
)
"""

_CREATE_DATASETS_TABLE = """
CREATE TABLE IF NOT EXISTS datasets (
    version             TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    total_samples       INTEGER DEFAULT 0,
    approved_samples    INTEGER DEFAULT 0,
    notes               TEXT DEFAULT '',
    locked              INTEGER DEFAULT 0    -- 1 = released, no changes allowed
)
"""

_CREATE_TRAINING_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS training_runs (
    run_id              TEXT PRIMARY KEY,
    dataset_version     TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    model_name          TEXT DEFAULT '',
    training_stage      TEXT DEFAULT '',
    n_samples           INTEGER DEFAULT 0,
    FOREIGN KEY (dataset_version) REFERENCES datasets(version)
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_samples_status   ON samples(status)",
    "CREATE INDEX IF NOT EXISTS idx_samples_language ON samples(detected_language)",
    "CREATE INDEX IF NOT EXISTS idx_samples_stage    ON samples(training_stage)",
    "CREATE INDEX IF NOT EXISTS idx_samples_quality  ON samples(quality_score)",
    "CREATE INDEX IF NOT EXISTS idx_samples_session  ON samples(session_id)",
]


# ─── Registry ─────────────────────────────────────────────────────────────────


class DatasetRegistry:
    """
    SQLite-backed registry of all pipeline samples and dataset versions.

    Usage:
        registry = DatasetRegistry(Path("data/pipeline/registry.db"))
        registry.register_sample(sample)
        registry.create_version("1.0.0", notes="First release")
        stats = registry.stats()

    Query examples:
        approved = registry.get_samples(status="approved", min_score=0.8)
        urdu     = registry.get_samples(language="ur")
        stage2   = registry.get_samples(training_stage="stage2_action")
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database connections.
        Commits on success, rolls back on exception.
        Thread-safe: creates new connection per call.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row  # access columns by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._conn() as conn:
            conn.execute(_CREATE_SAMPLES_TABLE)
            conn.execute(_CREATE_DATASETS_TABLE)
            conn.execute(_CREATE_TRAINING_RUNS_TABLE)
            for idx in _CREATE_INDEXES:
                conn.execute(idx)

        log.info("registry_initialized", path=str(self._db_path))

    # ── Sample Registration ────────────────────────────────────────────────

    def register_sample(self, sample: PipelineSample) -> None:
        """
        Insert or update a sample in the registry.
        Uses INSERT OR REPLACE so calling this multiple times is safe.
        """
        quality_score = sample.quality.quality_score if sample.quality else 0.0
        approved = int(
            sample.quality.approved_for_training if sample.quality else False
        )
        rejection = sample.quality.rejection_reason if sample.quality else ""
        language = sample.enriched.detected_language if sample.enriched else "unknown"
        action_type = sample.raw.action_type if sample.raw else ""

        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO samples (
                    sample_id, session_id, step_index, source_type, task,
                    task_category, status, quality_score, approved,
                    training_stage, detected_language, action_type,
                    schema_version, dataset_version, created_at, updated_at,
                    tags, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    sample.sample_id,
                    sample.session_id,
                    sample.step_index,
                    sample.source_type,
                    sample.task[:500],  # cap task length
                    sample.task_category,
                    sample.status,
                    quality_score,
                    approved,
                    sample.training_stage,
                    language,
                    action_type,
                    sample.schema_version,
                    sample.dataset_version,
                    sample.created_at,
                    sample.updated_at,
                    json.dumps(sample.tags),
                    sample.rejection_reason,
                ),
            )

    def register_batch(self, samples: list[PipelineSample]) -> int:
        """
        Bulk register samples. Returns count of registered samples.
        More efficient than calling register_sample() in a loop.
        """
        rows = []
        for sample in samples:
            quality_score = sample.quality.quality_score if sample.quality else 0.0
            approved = int(
                sample.quality.approved_for_training if sample.quality else False
            )
            rows.append(
                (
                    sample.sample_id,
                    sample.session_id,
                    sample.step_index,
                    sample.source_type,
                    sample.task[:500],
                    sample.task_category,
                    sample.status,
                    quality_score,
                    approved,
                    sample.training_stage,
                    sample.enriched.detected_language if sample.enriched else "unknown",
                    sample.raw.action_type if sample.raw else "",
                    sample.schema_version,
                    sample.dataset_version,
                    sample.created_at,
                    sample.updated_at,
                    json.dumps(sample.tags),
                    sample.quality.rejection_reason if sample.quality else "",
                )
            )

        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO samples (
                    sample_id, session_id, step_index, source_type, task,
                    task_category, status, quality_score, approved,
                    training_stage, detected_language, action_type,
                    schema_version, dataset_version, created_at, updated_at,
                    tags, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                rows,
            )

        return len(rows)

    # ── Querying ───────────────────────────────────────────────────────────

    def get_samples(
        self,
        status: str | None = None,
        min_score: float | None = None,
        language: str | None = None,
        training_stage: str | None = None,
        source_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Query samples with optional filters.
        Returns list of row dicts (lightweight — not full PipelineSample).
        """
        conditions: list[str] = []
        params: list = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if min_score is not None:
            conditions.append("quality_score >= ?")
            params.append(min_score)
        if language is not None:
            conditions.append("detected_language = ?")
            params.append(language)
        if training_stage is not None:
            conditions.append("training_stage = ?")
            params.append(training_stage)
        if source_type is not None:
            conditions.append("source_type = ?")
            params.append(source_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM samples {where} ORDER BY quality_score DESC {limit_clause}",
                params,
            ).fetchall()

        return [dict(row) for row in rows]

    # ── Dataset Versioning ─────────────────────────────────────────────────

    def create_version(
        self,
        version: str,
        notes: str = "",
        this_run_approved: int = 0,  # IMPROVISED: samples added in THIS run only
    ) -> None:
        """
        Create a new dataset version record.

        this_run_approved: count of samples approved in the current pipeline run.
        Stored as approved_samples for this version.
        The total_samples field reflects cumulative registry size.
        """
        v = DatasetVersion.parse(version)

        with self._conn() as conn:
            total_row = conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            total = total_row[0] if total_row else 0

            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (version, created_at, total_samples, approved_samples, notes)
                VALUES (?, ?, ?, ?, ?)
               """,
                (
                    str(v),
                    datetime.now(timezone.utc).isoformat(),
                    total,
                    this_run_approved,  # IMPROVISED: honest — only this run's count
                    notes,
                ),
            )

        log.info(
            "dataset_version_created",
            version=str(v),
            this_run_approved=this_run_approved,
            total_in_registry=total,
        )

    def lock_version(self, version: str) -> None:
        """
        Lock a version — no more changes allowed.
        Call this when releasing a version for training.
        """
        with self._conn() as conn:
            conn.execute(
                "UPDATE datasets SET locked = 1 WHERE version = ?",
                (version,),
            )
        log.info("dataset_version_locked", version=version)

    def list_versions(self) -> list[dict]:
        """List all dataset versions with their stats."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM datasets ORDER BY version DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    # ── Statistics ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Comprehensive statistics about the dataset.
        Returns a dict suitable for monitoring dashboards.
        """
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            approved = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE approved = 1"
            ).fetchone()[0]
            rejected = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE status = 'rejected'"
            ).fetchone()[0]
            quarantined = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE status = 'quarantined'"
            ).fetchone()[0]
            avg_score = (
                conn.execute(
                    "SELECT AVG(quality_score) FROM samples WHERE status = 'approved'"
                ).fetchone()[0]
                or 0.0
            )

            # Language distribution
            lang_rows = conn.execute("""
                SELECT detected_language, COUNT(*) as n
                FROM samples GROUP BY detected_language ORDER BY n DESC
            """).fetchall()

            # Stage distribution
            stage_rows = conn.execute("""
                SELECT training_stage, COUNT(*) as n
                FROM samples WHERE approved = 1
                GROUP BY training_stage ORDER BY n DESC
            """).fetchall()

            # Action type distribution
            action_rows = conn.execute("""
                SELECT action_type, COUNT(*) as n
                FROM samples GROUP BY action_type ORDER BY n DESC
            """).fetchall()

        return {
            "total_samples": total,
            "approved": approved,
            "rejected": rejected,
            "quarantined": quarantined,
            "pending": total - approved - rejected - quarantined,
            "approval_rate": round(approved / max(total, 1) * 100, 1),
            "avg_quality_score": round(avg_score, 4),
            "by_language": {r["detected_language"]: r["n"] for r in lang_rows},
            "by_stage": {r["training_stage"]: r["n"] for r in stage_rows},
            "by_action": {r["action_type"]: r["n"] for r in action_rows},
        }

    def contamination_check(
        self,
        test_sample_ids: list[str],
    ) -> list[str]:
        """
        Verify test set has zero overlap with training samples.
        Returns list of contaminated IDs (should be empty).

        Call this before every training run.
        """
        placeholders = ",".join("?" * len(test_sample_ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT sample_id FROM samples WHERE sample_id IN ({placeholders})",
                test_sample_ids,
            ).fetchall()
        return [row["sample_id"] for row in rows]
