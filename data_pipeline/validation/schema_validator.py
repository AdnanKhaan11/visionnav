"""
Gate 1 — Schema validation.
Fast structural check: required fields present, types correct.
"""

from __future__ import annotations

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample

_REQUIRED_FIELDS = [
    "sample_id",
    "session_id",
    "step_index",
    "source_type",
    "task",
]

_REQUIRED_RAW_FIELDS = [
    "screenshot_path",
    "action_type",
    "description",
]


def validate(sample: PipelineSample) -> None:
    """
    Validate top-level structure and raw content.
    Raises ValidationError (fatal) on any failure.
    """
    # ── Top-level fields ──────────────────────────────────────────────────
    for f in _REQUIRED_FIELDS:
        val = getattr(sample, f, None)
        if val is None or val == "":
            raise ValidationError(
                message=f"Required field '{f}' is missing or empty",
                sample_id=sample.sample_id,
                gate="schema",
                is_fatal=True,
            )

    # ── Step index must be non-negative ───────────────────────────────────
    if sample.step_index < 0:
        raise ValidationError(
            message=f"step_index must be >= 0, got {sample.step_index}",
            sample_id=sample.sample_id,
            gate="schema",
            is_fatal=True,
        )

    # ── Task must be a real instruction ───────────────────────────────────
    if len(sample.task.strip()) < 5:
        raise ValidationError(
            message=f"Task instruction too short: '{sample.task}'",
            sample_id=sample.sample_id,
            gate="schema",
            is_fatal=True,
        )

    # ── Raw content must be present ───────────────────────────────────────
    if sample.raw is None:
        raise ValidationError(
            message="sample.raw is None — no raw content to validate",
            sample_id=sample.sample_id,
            gate="schema",
            is_fatal=True,
        )

    # ── Required raw fields ───────────────────────────────────────────────
    for f in _REQUIRED_RAW_FIELDS:
        val = getattr(sample.raw, f, None)
        if val is None or val == "":
            raise ValidationError(
                message=f"Required raw field '{f}' is missing",
                sample_id=sample.sample_id,
                gate="schema",
                is_fatal=True,
            )

    # ── Action type must be a known value ─────────────────────────────────
    from visionnav.actions.schema import ActionType

    known = {a.value for a in ActionType}
    if sample.raw.action_type not in known:
        raise ValidationError(
            message=(
                f"Unknown action_type: '{sample.raw.action_type}'. "
                f"Valid: {sorted(known)}"
            ),
            sample_id=sample.sample_id,
            gate="schema",
            is_fatal=True,
        )
