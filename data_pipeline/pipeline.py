"""
Master pipeline orchestrator — ties all stages together.

This is the single entry point for the Dataset Factory.
One call processes recordings end-to-end:
  raw JSONL → validated → enriched → annotated → curated → exported

Architecture decision: synchronous (not async).
Why? The bottleneck is disk I/O and LLM API calls.
     asyncio adds complexity without meaningful speedup here.
     Stage 3 (1M+ samples/day) will use Celery workers instead.

Usage:
    pipeline = DatasetPipeline.from_config(Path("pipeline_config.json"))
    result   = pipeline.run(
        recordings_dir  = Path("data/recordings"),
        output_dir      = Path("data/training"),
        dataset_version = "1.0.0",
    )
    print(pipeline.reporter.format(result.metrics))
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from data_pipeline.annotation.difficulty_scorer import score as score_difficulty
from data_pipeline.core.exceptions import PipelineError
from data_pipeline.core.metadata import LineageStore, now_iso
from data_pipeline.core.schemas import SampleStatus
from data_pipeline.curation.quality_scorer import compute as compute_quality
from data_pipeline.curation.stage_router import route as route_stage
from data_pipeline.enrichment.enricher import SampleEnricher
from data_pipeline.exports.llamafactory import LlamaFactoryExporter
from data_pipeline.ingestion.sources.recorder_source import RecorderSource
from data_pipeline.monitoring.metrics import PipelineMetrics
from data_pipeline.monitoring.reporter import PipelineReporter
from data_pipeline.registry.registry import DatasetRegistry
from data_pipeline.validation.deduplicator import Deduplicator
from data_pipeline.validation.gate import ValidationGate
from data_pipeline.validation import (
    schema_validator,
    coordinate_validator,
    image_validator,
)

log = structlog.get_logger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for one pipeline run.
    All paths and thresholds in one place.
    Change here — nothing else changes.
    """

    # Paths
    pipeline_dir: Path = Path("data/pipeline")

    # Quality thresholds
    quality_threshold: float = 0.65
    image_min_size_kb: float = 10.0

    # Annotation
    annotate_samples: bool = False  # False in Stage 1 (no API key needed)
    annotation_api_key: str = ""
    annotation_model: str = "claude-haiku-4-5-20251001"

    # Export
    export_stages: list[str] = field(
        default_factory=lambda: [
            "stage1_grounding",
            "stage2_action",
            "stage3_planning",
        ]
    )

    @property
    def registry_path(self) -> Path:
        return self.pipeline_dir / "registry.db"

    @property
    def lineage_path(self) -> Path:
        return self.pipeline_dir / "lineage.jsonl"

    @property
    def dedup_index_path(self) -> Path:
        return self.pipeline_dir / ".dedup_index.json"

    @property
    def reports_dir(self) -> Path:
        return self.pipeline_dir / "reports"

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                val = Path(v) if k.endswith("_dir") or k.endswith("_path") else v
                setattr(config, k, val)
        return config


@dataclass
class PipelineResult:
    """
    Complete result of one pipeline run.
    Contains everything needed for analysis and reporting.
    """

    run_id: str
    metrics: PipelineMetrics
    approved_count: int
    rejected_count: int
    exported: dict[str, int]  # stage → count
    output_dir: Path
    dataset_version: str


class DatasetPipeline:
    """
    Orchestrates the complete Dataset Factory pipeline.

    Stage-by-stage processing with metrics tracking at every step.
    Designed to be run repeatedly as new recordings are added.
    Idempotent: running on the same files twice produces same results
    (dedup index prevents double-counting).
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._config.pipeline_dir.mkdir(parents=True, exist_ok=True)
        self._config.reports_dir.mkdir(parents=True, exist_ok=True)

        # ── Build all components ───────────────────────────────────────────
        self._lineage = LineageStore(self._config.lineage_path)
        self._registry = DatasetRegistry(self._config.registry_path)

        # Validation gate — build once, reuse for all samples
        dedup = Deduplicator(self._config.dedup_index_path)
        self._gate = ValidationGate(lineage_store=self._lineage)
        self._gate.register("schema", schema_validator.validate, is_fatal=True)
        self._gate.register("coordinates", coordinate_validator.validate, is_fatal=True)
        self._gate.register("image", image_validator.validate, is_fatal=False)
        self._gate.register("dedup", dedup.validate, is_fatal=True)

        self._enricher = SampleEnricher(lineage_store=self._lineage)
        self._reporter = PipelineReporter()

        # Annotator is optional — only built if API key provided
        self._annotator = None
        if self._config.annotate_samples and self._config.annotation_api_key:
            from data_pipeline.annotation.auto_annotator import AutoAnnotator

            self._annotator = AutoAnnotator(
                api_key=self._config.annotation_api_key,
                model=self._config.annotation_model,
            )

    @property
    def reporter(self) -> PipelineReporter:
        return self._reporter

    def run(
        self,
        recordings_dir: Path,
        output_dir: Path,
        dataset_version: str = "1.0.0",
    ) -> PipelineResult:
        """
        Run the complete pipeline on all recordings in a directory.

        Steps:
          1. Ingest all session_*.jsonl files
          2. Validate each sample through all gates
          3. Enrich passing samples (OCR, language, hashes)
          4. Score difficulty (rule-based, free)
          5. Auto-annotate if API key configured
          6. Compute quality score
          7. Route to training stage
          8. Register in dataset registry
          9. Export approved samples to LLaMA-Factory format
          10. Generate and save run report

        Args:
            recordings_dir:  directory containing session_*.jsonl files
            output_dir:      where to write training JSONL + dataset_info
            dataset_version: semantic version string "MAJOR.MINOR.PATCH"

        Returns:
            PipelineResult with full statistics
        """
        run_id = str(uuid.uuid4())[:8]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = PipelineMetrics(run_id=run_id)

        log.info(
            "pipeline_started",
            run_id=run_id,
            recordings_dir=str(recordings_dir),
            output_dir=str(output_dir),
            dataset_version=dataset_version,
            annotate=self._config.annotate_samples,
        )

        # ── STAGE 1: INGESTION ────────────────────────────────────────────
        source = RecorderSource(lineage_store=self._lineage)
        ingest_result = source.ingest_directory(recordings_dir)

        metrics.ingestion.processed = ingest_result.total_lines
        metrics.ingestion.passed = ingest_result.ingested
        metrics.ingestion.failed = ingest_result.failed

        approved_samples = []

        # ── STAGES 2-7: per-sample processing ────────────────────────────
        for sample in ingest_result.samples:

            # STAGE 2: VALIDATION
            gate_result = self._gate.run(sample)
            metrics.validation.processed += 1

            if gate_result.passed:
                metrics.validation.passed += 1
            else:
                metrics.validation.failed += 1
                metrics.record_rejection(gate_result.failed_gate or gate_result.status)
                continue  # skip to next sample

            # STAGE 3: ENRICHMENT
            enrich_report = self._enricher.enrich(sample)
            metrics.enrichment.processed += 1

            if enrich_report.failed:
                metrics.enrichment.failed += 1
            else:
                metrics.enrichment.passed += 1

            # Track language after enrichment
            if sample.enriched:
                metrics.record_language(sample.enriched.detected_language)

            # STAGE 4: DIFFICULTY SCORING (always runs, free)
            score_difficulty(sample)

            # STAGE 5: ANNOTATION (only if annotator configured)
            metrics.annotation.processed += 1
            if self._annotator:
                ann_result = self._annotator.annotate(sample)
                if ann_result.success:
                    metrics.annotation.passed += 1
                else:
                    metrics.annotation.failed += 1
                    log.warning(
                        "annotation_failed",
                        sample_id=sample.sample_id,
                        error=ann_result.error,
                    )
            else:
                # No annotator — skip but count as passed
                metrics.annotation.passed += 1

            # STAGE 6: QUALITY SCORING
            quality = compute_quality(
                sample,
                quality_threshold=self._config.quality_threshold,
            )
            metrics.curation.processed += 1
            metrics.record_quality(quality)

            if sample.quality and sample.quality.approved_for_training:
                metrics.curation.passed += 1
            else:
                metrics.curation.failed += 1
                reason = (
                    sample.quality.rejection_reason if sample.quality else "low_quality"
                )
                metrics.record_rejection(reason)
                continue

            # STAGE 7: STAGE ROUTING
            stage = route_stage(sample)
            if stage:
                metrics.record_stage_route(stage)
                sample.dataset_version = dataset_version
                approved_samples.append(sample)
            else:
                metrics.record_rejection("no_stage_assigned")

        # ── REGISTER all approved samples ─────────────────────────────────
        if approved_samples:
            registered = self._registry.register_batch(approved_samples)
            log.info("samples_registered", count=registered)

        self._registry.create_version(
            dataset_version,
            notes=f"Run {run_id} — {len(approved_samples)} approved samples",
        )

        # ── EXPORT ────────────────────────────────────────────────────────
        exporter = LlamaFactoryExporter(output_dir)
        export_results = exporter.export_all_stages(
            samples=approved_samples,
            dataset_version=dataset_version,
        )

        exported_counts: dict[str, int] = {}
        for stage_name, result in export_results.items():
            metrics.exported_per_stage[stage_name] = result.exported
            metrics.export.processed += result.total_samples
            metrics.export.passed += result.exported
            exported_counts[stage_name] = result.exported

        # ── FINALIZE METRICS ──────────────────────────────────────────────
        metrics.finish()

        # ── SAVE REPORT ───────────────────────────────────────────────────
        report_path = self._config.reports_dir / f"run_{run_id}.json"
        self._reporter.save(metrics, report_path)

        log.info(
            "pipeline_complete",
            run_id=run_id,
            approved=len(approved_samples),
            rejected=metrics.validation.failed + metrics.curation.failed,
            exported=sum(exported_counts.values()),
        )

        return PipelineResult(
            run_id=run_id,
            metrics=metrics,
            approved_count=len(approved_samples),
            rejected_count=metrics.validation.failed + metrics.curation.failed,
            exported=exported_counts,
            output_dir=output_dir,
            dataset_version=dataset_version,
        )
