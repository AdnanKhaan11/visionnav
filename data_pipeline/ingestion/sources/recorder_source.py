"""
ActionRecorder source — ingests trajectories recorded by Assignment 5 system.

This is the primary data source for Stage 1.
Reads session_*.jsonl files produced by ActionRecorder,
normalizes each step to PipelineSample,
and creates SampleLineage for traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import structlog

from data_pipeline.core.exceptions import IngestionError
from data_pipeline.core.metadata import (
    SampleLineage,
    LineageStore,
    new_sample_id,
    now_iso,
)
from data_pipeline.core.schemas import PipelineSample, SourceType
from data_pipeline.ingestion.reader import stream_jsonl
from data_pipeline.ingestion.normalizer import normalize_recorded_step

log = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    """Summary of one ingestion run."""

    source_path: str
    total_lines: int
    ingested: int
    failed: int
    samples: list[PipelineSample]

    @property
    def success_rate(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return self.ingested / self.total_lines


class RecorderSource:
    """
    Reads ActionRecorder JSONL files and produces PipelineSamples.

    Usage:
        source  = RecorderSource(lineage_store)
        samples = source.ingest_file(Path("data/recordings/session_abc123.jsonl"))
        for sample in samples:
            validation_gate.run(sample)

    Batch ingestion:
        results = source.ingest_directory(Path("data/recordings/"))
    """

    def __init__(
        self,
        lineage_store: LineageStore,
        collector_id: str = "system",
        device_name: str = "unknown",
    ) -> None:
        self._lineage = lineage_store
        self._collector_id = collector_id
        self._device_name = device_name

    def ingest_file(self, path: Path) -> Iterator[PipelineSample]:
        """
        Ingest one session JSONL file.
        Yields PipelineSample for each valid step.
        Creates SampleLineage for every sample.

        Args:
            path: path to session_*.jsonl file

        Yields:
            PipelineSample ready for validation
        """
        path = Path(path)
        log.info("ingesting_file", path=str(path))

        for raw_line in stream_jsonl(path):
            try:
                sample = normalize_recorded_step(
                    data=raw_line.data,
                    source_file=raw_line.source_file,
                    line_number=raw_line.line_number,
                    source_type=SourceType.HUMAN_DEMO.value,
                )

                # Create and persist lineage record
                lineage = SampleLineage(
                    sample_id=sample.sample_id,
                    source_file=raw_line.source_file,
                    source_line=raw_line.line_number,
                    collector_id=self._collector_id,
                    collection_device=self._device_name,
                )
                lineage.add_event(
                    event="ingested",
                    stage="recorder_source",
                    result="success",
                    session=sample.session_id,
                    step=sample.step_index,
                )
                self._lineage.save(lineage)

                log.debug(
                    "sample_ingested",
                    sample_id=sample.sample_id,
                    session_id=sample.session_id,
                    step=sample.step_index,
                    action=sample.raw.action_type if sample.raw else "?",
                )

                yield sample

            except IngestionError as exc:
                log.error(
                    "ingestion_failed",
                    file=str(path),
                    line=raw_line.line_number,
                    error=str(exc),
                )
                continue  # skip bad line, continue with rest of file

    def ingest_directory(
        self,
        directory: Path,
        pattern: str = "session_*.jsonl",
    ) -> IngestionResult:
        """
        Ingest all session files in a directory.
        Returns a summary with all samples and statistics.

        Args:
            directory: directory containing session_*.jsonl files
            pattern:   glob pattern to match recording files

        Returns:
            IngestionResult with all samples and summary stats
        """
        directory = Path(directory)
        files = sorted(directory.glob(pattern))

        if not files:
            log.warning(
                "no_recordings_found", directory=str(directory), pattern=pattern
            )
            return IngestionResult(
                source_path=str(directory),
                total_lines=0,
                ingested=0,
                failed=0,
                samples=[],
            )

        log.info(
            "batch_ingestion_started",
            directory=str(directory),
            files=len(files),
        )

        all_samples: list[PipelineSample] = []
        total_lines = ingested = failed = 0

        for file_path in files:
            for sample in self.ingest_file(file_path):
                all_samples.append(sample)
                ingested += 1
            total_lines += sum(1 for _ in open(file_path))  # count lines

        failed = total_lines - ingested

        log.info(
            "batch_ingestion_complete",
            total_lines=total_lines,
            ingested=ingested,
            failed=failed,
            success_rate=f"{ingested/max(total_lines,1)*100:.1f}%",
        )

        return IngestionResult(
            source_path=str(directory),
            total_lines=total_lines,
            ingested=ingested,
            failed=failed,
            samples=all_samples,
        )
