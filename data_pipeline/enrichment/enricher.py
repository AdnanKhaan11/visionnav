"""
Enrichment orchestrator — runs all enrichers on a sample in the correct order.

Order matters:
  1. hash_computer  first (hashes needed by deduplicator later)
  2. ocr_enricher   second (OCR needed by language_detector)
  3. language_detector last (needs OCR text from ocr_enricher)

Each enricher is called with the sample.
Each enricher modifies sample.enriched in place.
If an enricher fails, we log and continue with partial enrichment.
We never crash the pipeline because one enricher had an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from data_pipeline.core.exceptions import EnrichmentError
from data_pipeline.core.metadata import LineageStore, now_iso
from data_pipeline.core.schemas import (
    EnrichedContent,
    PipelineSample,
    SampleStatus,
)
from data_pipeline.enrichment import (
    hash_computer,
    ocr_enricher,
    language_detector,
)

log = structlog.get_logger(__name__)


@dataclass
class EnrichmentReport:
    """Summary of enrichment results for one sample."""

    sample_id: str
    completed: list[str] = field(default_factory=list)  # enrichers that succeeded
    failed: list[str] = field(default_factory=list)  # enrichers that failed
    warnings: list[str] = field(default_factory=list)


class SampleEnricher:
    """
    Runs all enrichment steps on a sample.

    Each step is run independently — failure in one does not
    prevent the others from running. The sample gets whatever
    enrichment is possible.

    Usage:
        enricher = SampleEnricher(lineage_store)
        report   = enricher.enrich(sample)
        if report.failed:
            log.warning("partial_enrichment", sample_id=sample.sample_id)
    """

    # Ordered list of (name, module) pairs
    # Order is intentional — see module docstring
    _PIPELINE = [
        ("hash_computer", hash_computer),
        ("ocr_enricher", ocr_enricher),
        ("language_detector", language_detector),
    ]

    def __init__(
        self,
        lineage_store: LineageStore | None = None,
    ) -> None:
        self._lineage = lineage_store

    def enrich(self, sample: PipelineSample) -> EnrichmentReport:
        """
        Run all enrichers on one sample.

        Updates sample.enriched in place.
        Updates sample.status to ENRICHED on success.
        Updates lineage with enrichment events.

        Returns:
            EnrichmentReport with success/failure details
        """
        report = EnrichmentReport(sample_id=sample.sample_id)

        # Ensure enriched container exists
        if sample.enriched is None:
            sample.enriched = EnrichedContent()

        sample.status = SampleStatus.ENRICHING.value

        for name, module in self._PIPELINE:
            try:
                module.enrich(sample)
                report.completed.append(name)
                self._record_event(sample, name, "success")

                log.debug(
                    "enricher_complete",
                    sample_id=sample.sample_id,
                    enricher=name,
                )

            except EnrichmentError as exc:
                report.failed.append(name)
                report.warnings.append(f"{name}: {exc.message}")
                self._record_event(sample, name, "failed", error=exc.message)

                log.warning(
                    "enricher_failed",
                    sample_id=sample.sample_id,
                    enricher=name,
                    error=exc.message,
                    retryable=exc.retryable,
                )
                # Continue to next enricher even after failure

            except Exception as exc:
                report.failed.append(name)
                report.warnings.append(f"{name}: unexpected error: {exc}")
                self._record_event(sample, name, "error", error=str(exc))

                log.error(
                    "enricher_unexpected_error",
                    sample_id=sample.sample_id,
                    enricher=name,
                    error=str(exc),
                    exc_info=True,
                )

        # Mark enrichment complete (even if some enrichers failed)
        sample.status = SampleStatus.ENRICHED.value
        sample.updated_at = now_iso()

        log.info(
            "enrichment_complete",
            sample_id=sample.sample_id,
            completed=len(report.completed),
            failed=len(report.failed),
        )

        return report

    def enrich_batch(
        self,
        samples: list[PipelineSample],
    ) -> list[EnrichmentReport]:
        """
        Enrich a batch of samples sequentially.
        Returns list of reports in same order as input.
        """
        reports = []
        total = len(samples)

        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                log.info(
                    "enrichment_progress",
                    processed=i + 1,
                    total=total,
                )
            report = self.enrich(sample)
            reports.append(report)

        return reports

    def _record_event(
        self,
        sample: PipelineSample,
        enricher: str,
        result: str,
        **details,
    ) -> None:
        if self._lineage:
            self._lineage.add_event(
                sample_id=sample.sample_id,
                event="enriched",
                stage=enricher,
                result=result,
                **details,
            )
