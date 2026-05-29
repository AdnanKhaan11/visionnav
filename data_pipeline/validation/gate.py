"""
ValidationGate — orchestrates all validation checks for one sample.

The gate runs validators in order.
Early cheap checks run first (schema, coordinates).
Expensive checks run later (image quality, deduplication).

If a sample fails any gate:
  - is_fatal=True  → REJECTED (will never be used for training)
  - is_fatal=False → QUARANTINED (human review needed)

Why separation?
  REJECTED: coordinates [1.5, 0.5] can never be fixed.
  QUARANTINED: low OCR confidence might be fixable with better OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import structlog

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample, QualityMetrics, SampleStatus
from data_pipeline.core.metadata import LineageStore

log = structlog.get_logger(__name__)


@dataclass
class GateResult:
    """Result of running the full validation gate on one sample."""

    sample_id: str
    passed: bool
    status: str  # APPROVED → continue, QUARANTINED, REJECTED
    failed_gate: str = ""  # which validator failed
    reason: str = ""  # human-readable explanation
    warnings: list[str] = field(default_factory=list)


class ValidationGate:
    """
    Runs all validators on a sample and returns a decision.

    Validators are registered with:
      - name:      identifier for logging and lineage
      - fn:        the validation function (sample → raises ValidationError or passes)
      - is_fatal:  if True and fn raises → REJECT; if False → QUARANTINE
      - enabled:   can be disabled without removing from code

    Usage:
        gate = ValidationGate()
        gate.register("schema",      schema_validator.validate,      is_fatal=True)
        gate.register("coordinates", coordinate_validator.validate,  is_fatal=True)
        gate.register("image",       image_validator.validate,       is_fatal=False)
        gate.register("dedup",       deduplicator.validate,          is_fatal=True)

        result = gate.run(sample)
    """

    def __init__(self, lineage_store: LineageStore | None = None) -> None:
        self._validators: list[dict] = []
        self._lineage = lineage_store

    def register(
        self,
        name: str,
        fn: Callable[[PipelineSample], None],
        is_fatal: bool = False,
        enabled: bool = True,
    ) -> None:
        """
        Register a validator function.

        fn must:
          - accept one PipelineSample argument
          - raise ValidationError if validation fails
          - return None if validation passes (no return value)
        """
        self._validators.append(
            {
                "name": name,
                "fn": fn,
                "is_fatal": is_fatal,
                "enabled": enabled,
            }
        )

    def run(self, sample: PipelineSample) -> GateResult:
        """
        Run all registered validators on the sample.

        Returns GateResult with pass/fail decision.
        Updates sample.status and sample.quality in place.
        """
        if sample.quality is None:
            sample.quality = QualityMetrics()

        warnings: list[str] = []

        for validator in self._validators:
            if not validator["enabled"]:
                continue

            name = validator["name"]
            fn = validator["fn"]
            is_fatal = validator["is_fatal"]

            try:
                fn(sample)

                # Validator passed — record in lineage
                self._record_event(sample.sample_id, name, "pass")
                log.debug(
                    "gate_pass",
                    sample_id=sample.sample_id,
                    gate=name,
                )

            except ValidationError as exc:
                self._record_event(
                    sample.sample_id,
                    name,
                    "fail",
                    reason=exc.message,
                    fatal=is_fatal,
                )

                log.warning(
                    "gate_fail",
                    sample_id=sample.sample_id,
                    gate=name,
                    reason=exc.message,
                    fatal=is_fatal,
                )

                if is_fatal:
                    sample.status = SampleStatus.REJECTED.value
                    if sample.quality:
                        sample.quality.rejection_reason = f"{name}: {exc.message}"
                    return GateResult(
                        sample_id=sample.sample_id,
                        passed=False,
                        status=SampleStatus.REJECTED.value,
                        failed_gate=name,
                        reason=exc.message,
                        warnings=warnings,
                    )
                else:
                    # Non-fatal → quarantine but continue recording warnings
                    warnings.append(f"{name}: {exc.message}")
                    sample.status = SampleStatus.QUARANTINED.value

        # All validators ran
        if sample.status == SampleStatus.QUARANTINED.value:
            return GateResult(
                sample_id=sample.sample_id,
                passed=False,
                status=SampleStatus.QUARANTINED.value,
                reason="; ".join(warnings),
                warnings=warnings,
            )

        # All passed
        sample.status = SampleStatus.ENRICHING.value
        if sample.quality:
            sample.quality.schema_valid = True
        return GateResult(
            sample_id=sample.sample_id,
            passed=True,
            status=SampleStatus.ENRICHING.value,
            warnings=warnings,
        )

    def _record_event(
        self,
        sample_id: str,
        gate: str,
        result: str,
        **details,
    ) -> None:
        if self._lineage:
            self._lineage.add_event(
                sample_id=sample_id,
                event="validated",
                stage=gate,
                result=result,
                **details,
            )
