"""
Quality scorer — computes a composite 0.0-1.0 quality score.

The score is a weighted average of multiple quality dimensions.
Each dimension tests a different aspect of training value.

Score interpretation:
  0.90 - 1.00: Excellent — use in all training stages
  0.80 - 0.90: Good — use in all training stages
  0.70 - 0.80: Acceptable — use with caution
  0.60 - 0.70: Marginal — review recommended
  0.00 - 0.60: Poor — reject from training

The threshold for training approval is configurable.
Default: 0.65 (Stage 1). Raise to 0.75 as dataset grows.
"""

from __future__ import annotations

import structlog

from data_pipeline.core.schemas import PipelineSample, QualityMetrics, SampleStatus
from data_pipeline.core.metadata import now_iso

log = structlog.get_logger(__name__)


# ── Scoring weights ────────────────────────────────────────────────────────────
# Must sum to 1.0
_WEIGHTS = {
    "validation": 0.30,  # schema, image, coordinates all valid
    "reasoning": 0.35,  # has reasoning, length, verified
    "ocr_quality": 0.20,  # OCR confidence and coverage
    "diversity": 0.15,  # action type, screen content uniqueness
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"


def compute(
    sample: PipelineSample,
    quality_threshold: float = 0.65,
) -> float:
    """
    Compute composite quality score and update sample.quality.

    Args:
        sample:            sample with validation + enrichment + annotation
        quality_threshold: minimum score for training approval

    Returns:
        Quality score between 0.0 and 1.0
    """
    if sample.quality is None:
        sample.quality = QualityMetrics()

    # Compute each dimension
    validation_score = _score_validation(sample)
    reasoning_score = _score_reasoning(sample)
    ocr_score = _score_ocr_quality(sample)
    diversity_score = _score_diversity(sample)

    # Weighted composite
    composite = (
        _WEIGHTS["validation"] * validation_score
        + _WEIGHTS["reasoning"] * reasoning_score
        + _WEIGHTS["ocr_quality"] * ocr_score
        + _WEIGHTS["diversity"] * diversity_score
    )

    composite = max(0.0, min(1.0, composite))

    # Update quality object
    sample.quality.quality_score = round(composite, 4)
    sample.quality.approved_for_training = composite >= quality_threshold
    sample.updated_at = now_iso()

    if composite >= quality_threshold:
        sample.status = SampleStatus.APPROVED.value
    else:
        sample.status = SampleStatus.REJECTED.value
        sample.quality.rejection_reason = (
            f"Quality score {composite:.3f} below threshold {quality_threshold}"
        )

    log.debug(
        "quality_scored",
        sample_id=sample.sample_id,
        score=round(composite, 4),
        approved=sample.quality.approved_for_training,
        breakdown={
            "validation": round(validation_score, 3),
            "reasoning": round(reasoning_score, 3),
            "ocr": round(ocr_score, 3),
            "diversity": round(diversity_score, 3),
        },
    )

    return composite


def _score_validation(sample: PipelineSample) -> float:
    """
    0.0 to 1.0 — did the sample pass all validation gates?
    Full score only if ALL gates passed.
    """
    q = sample.quality
    if q is None:
        return 0.0

    checks = [
        q.schema_valid,
        q.image_valid,
        q.coordinates_valid,
        q.no_pii_detected,
        q.not_duplicate,
    ]

    # All checks must pass for full score
    # Each failure reduces score proportionally
    passed = sum(1 for c in checks if c)
    return passed / len(checks)


def _score_reasoning(sample: PipelineSample) -> float:
    """
    0.0 to 1.0 — how good is the reasoning annotation?

    Two tiers:
      Tier 1 (partial score): raw description exists — basic signal present
      Tier 2 (full score):    LLM annotation exists — full reasoning chain
    """
    # ── Tier 1: raw description gives partial credit ───────────────────────
    # raw.description is written by the recorder ("Click Submit button")
    # It is not chain-of-thought, but it IS label information.
    # A model can learn from it even without full LLM annotation.
    description_score = 0.0
    if (
        sample.raw
        and sample.raw.description
        and len(sample.raw.description.strip()) > 5
    ):
        description_score = 0.25  # partial credit for having any label

    # ── Tier 2: LLM annotation adds to description score ──────────────────
    ann = sample.annotation
    if ann is None or not ann.reasoning:
        return description_score  # only partial credit

    score = description_score  # starts with description credit

    # Has reasoning (detailed chain-of-thought)
    if len(ann.reasoning.strip()) > 10:
        score += 0.25  # credit for having reasoning

    # Reasoning length — longer = more detailed = better signal
    length = len(ann.reasoning)
    if length >= 200:
        score += 0.25
    elif length >= 100:
        score += 0.15
    elif length >= 50:
        score += 0.08

    # Verified against OCR — elements mentioned actually exist on screen
    if ann.reasoning_verified:
        score += 0.15

    # Has intent description
    if ann.intent and len(ann.intent.strip()) > 5:
        score += 0.10

    return min(1.0, score)


def _score_ocr_quality(sample: PipelineSample) -> float:
    """
    0.0 to 1.0 — how good is the OCR data?
    """
    enriched = sample.enriched
    if enriched is None:
        return 0.0

    score = 0.0

    # Has OCR regions at all
    if enriched.n_ocr_regions > 0:
        score += 0.4

    # Reasonable number of regions (not too few, not blank)
    if enriched.n_ocr_regions >= 5:
        score += 0.3
    elif enriched.n_ocr_regions >= 2:
        score += 0.1

    # Average confidence from detailed regions
    if enriched.ocr_regions_detailed:
        avg_conf = sum(
            r.get("confidence", 0.0) for r in enriched.ocr_regions_detailed
        ) / len(enriched.ocr_regions_detailed)

        if avg_conf >= 0.85:
            score += 0.3
        elif avg_conf >= 0.70:
            score += 0.2
        elif avg_conf >= 0.50:
            score += 0.1

    return min(1.0, score)


def _score_diversity(sample: PipelineSample) -> float:
    """
    0.0 to 1.0 — does this sample add diversity to the dataset?

    Higher diversity = more training value.
    Multilingual samples get bonus points.
    Error recovery samples get bonus points.
    """
    score = 0.5  # base score — assume some diversity

    # Multilingual samples are rare and valuable
    if sample.enriched:
        lang = sample.enriched.detected_language
        if lang in ("ur", "ps", "ar", "hi"):
            score += 0.3

    # Error recovery samples are explicitly rare
    if sample.annotation and sample.annotation.is_error_step:
        score += 0.2

    # Tagged samples (well-categorized) are more usable
    if len(sample.tags) >= 3:
        score += 0.1

    # Complex actions (not just clicks) add diversity
    if sample.raw:
        action_type = sample.raw.action_type
        if action_type in ("type", "right_click", "key", "scroll"):
            score += 0.1

    return min(1.0, score)
