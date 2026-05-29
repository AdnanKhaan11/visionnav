"""
Dataset Factory pipeline exceptions.

Every pipeline failure is typed.
Typed exceptions let callers handle specific failures differently:
  - ValidationError → quarantine sample, continue
  - IngestionError  → log and skip, continue
  - EnrichmentError → retry, then quarantine
  - CriticalError   → stop pipeline entirely
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all dataset pipeline errors."""

    def __init__(self, message: str, sample_id: str = "") -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.message = message

    def __str__(self) -> str:
        if self.sample_id:
            return f"[sample={self.sample_id}] {self.message}"
        return self.message


class IngestionError(PipelineError):
    """Failed to read or parse a raw recording file."""


class ValidationError(PipelineError):
    """Sample failed a quality gate check."""

    def __init__(
        self,
        message: str,
        sample_id: str = "",
        gate: str = "",
        is_fatal: bool = False,
    ) -> None:
        super().__init__(message, sample_id)
        self.gate = gate  # which gate failed: "schema", "image", etc.
        self.is_fatal = is_fatal  # if True → reject, if False → quarantine


class EnrichmentError(PipelineError):
    """An enrichment step failed (OCR, language detection, etc.)."""

    def __init__(
        self,
        message: str,
        sample_id: str = "",
        enricher: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, sample_id)
        self.enricher = enricher
        self.retryable = retryable


class AnnotationError(PipelineError):
    """Auto-annotation failed (LLM error, timeout, etc.)."""


class RegistryError(PipelineError):
    """Dataset registry operation failed."""


class ExportError(PipelineError):
    """Export to training format failed."""


class SchemaVersionError(PipelineError):
    """Sample schema version is incompatible with current pipeline."""
