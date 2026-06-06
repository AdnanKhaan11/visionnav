"""
Stage router — assigns each sample to the correct training stage.

Three training stages in LLaMA-Factory:
  Stage 1 (grounding):  learn to locate elements on screen
  Stage 2 (action):     learn to predict the correct action
  Stage 3 (planning):   learn multi-step reasoning

Not all samples are useful for all stages.
Routing ensures each stage gets the right training signal.
"""

from __future__ import annotations

import structlog

from data_pipeline.core.schemas import (
    PipelineSample,
    TrainingStage,
    SampleStatus,
)

log = structlog.get_logger(__name__)

# Minimum quality score per stage (higher stages need better data)
_STAGE_QUALITY_THRESHOLDS = {
    TrainingStage.STAGE1_GROUNDING: 0.60,
    TrainingStage.STAGE2_ACTION: 0.62,
    TrainingStage.STAGE3_PLANNING: 0.80,
}

# Action types relevant for grounding training
_GROUNDING_ACTIONS = {"click", "double_click", "right_click"}

# Minimum step index for planning samples (must be part of longer trajectory)
_PLANNING_MIN_STEP = 5


def route(sample: PipelineSample) -> str:
    score = sample.quality_score()
    action = sample.raw.action_type if sample.raw else ""

    # ── Stage 3 ───────────────────────────────────────────────────────────
    threshold_3 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE3_PLANNING]

    is_long_horizon = sample.step_index >= _PLANNING_MIN_STEP

    has_good_reasoning = (
        sample.annotation is not None
        and len(sample.annotation.reasoning) >= 150
        and sample.annotation.reasoning_verified
    )

    has_intent = sample.annotation is not None and len(sample.annotation.intent) > 10

    if score >= threshold_3 and is_long_horizon and has_good_reasoning and has_intent:
        sample.training_stage = TrainingStage.STAGE3_PLANNING.value

        log.debug(
            "stage_routed",
            sample_id=sample.sample_id,
            stage=sample.training_stage,
        )

        return sample.training_stage

    # ── Stage 2 ───────────────────────────────────────────────────────────
    # Accepts ALL action types (click, type, key, scroll, done)
    # Requires either:
    #   1. Valid annotation reasoning
    #   2. Or a meaningful raw description

    threshold_2 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE2_ACTION]

    has_any_context = (
        sample.annotation is not None and len(sample.annotation.reasoning.strip()) >= 20
    ) or (sample.raw is not None and len((sample.raw.description or "").strip()) >= 5)

    if score >= threshold_2 and has_any_context:
        sample.training_stage = TrainingStage.STAGE2_ACTION.value

        log.debug(
            "stage_routed",
            sample_id=sample.sample_id,
            stage=sample.training_stage,
        )

        return sample.training_stage

    # ── Stage 1 ───────────────────────────────────────────────────────────
    # Accepts click actions with coordinates and OCR
    # Stage 1 is intentionally unreachable when all samples have descriptions.
    # Reserved for future curriculum training (coordinate-only prediction without reasoning).
    # To activate: add explicit bypass condition before Stage 2 for description-less click samples.
    threshold_1 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE1_GROUNDING]

    has_coordinates = (
        sample.raw is not None
        and sample.raw.coordinates is not None
        and action in _GROUNDING_ACTIONS
    )

    has_ocr = sample.enriched is not None and sample.enriched.n_ocr_regions >= 2

    if score >= threshold_1 and has_coordinates and has_ocr:
        sample.training_stage = TrainingStage.STAGE1_GROUNDING.value

        log.debug(
            "stage_routed",
            sample_id=sample.sample_id,
            stage=sample.training_stage,
        )

        return sample.training_stage

    # ── No stage ──────────────────────────────────────────────────────────

    log.debug(
        "stage_route_failed",
        sample_id=sample.sample_id,
        score=score,
        action=action,
    )

    sample.training_stage = ""
    return ""
