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
    TrainingStage.STAGE2_ACTION: 0.70,
    TrainingStage.STAGE3_PLANNING: 0.80,
}

# Action types relevant for grounding training
_GROUNDING_ACTIONS = {"click", "double_click", "right_click"}

# Minimum step index for planning samples (must be part of longer trajectory)
_PLANNING_MIN_STEP = 5


def route(sample: PipelineSample) -> str:
    """
    Determine which training stage this sample belongs to.

    Updates sample.training_stage in place.

    Returns:
        TrainingStage enum value string
    """
    score = sample.quality_score()
    action = sample.raw.action_type if sample.raw else ""

    # ── Stage 3: Planning ─────────────────────────────────────────────────
    # Long trajectories with detailed reasoning
    threshold_3 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE3_PLANNING]

    is_long_horizon = sample.step_index >= _PLANNING_MIN_STEP
    has_good_reasoning = (
        sample.annotation is not None
        and len(sample.annotation.reasoning) >= 150
        and sample.annotation.reasoning_verified
    )
    has_intent = sample.annotation is not None and len(sample.annotation.intent) > 10

    if score >= threshold_3 and is_long_horizon and has_good_reasoning and has_intent:
        stage = TrainingStage.STAGE3_PLANNING.value
        log.debug("stage_routed", sample_id=sample.sample_id, stage="stage3")
        sample.training_stage = stage
        return stage

    # ── Stage 2: Action prediction ─────────────────────────────────────────
    # Samples with clear task context and good quality
    threshold_2 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE2_ACTION]

    has_reasoning = (
        sample.annotation is not None and len(sample.annotation.reasoning) >= 50
    )

    if score >= threshold_2 and has_reasoning:
        stage = TrainingStage.STAGE2_ACTION.value
        log.debug("stage_routed", sample_id=sample.sample_id, stage="stage2")
        sample.training_stage = stage
        return stage

    # ── Stage 1: Element grounding ─────────────────────────────────────────
    # Click samples with coordinates and OCR text
    threshold_1 = _STAGE_QUALITY_THRESHOLDS[TrainingStage.STAGE1_GROUNDING]

    has_coordinates = (
        sample.raw is not None
        and sample.raw.coordinates is not None
        and action in _GROUNDING_ACTIONS
    )
    has_ocr = sample.enriched is not None and sample.enriched.n_ocr_regions >= 2

    if score >= threshold_1 and has_coordinates and has_ocr:
        stage = TrainingStage.STAGE1_GROUNDING.value
        log.debug("stage_routed", sample_id=sample.sample_id, stage="stage1")
        sample.training_stage = stage
        return stage

    # ── No stage — below all thresholds ───────────────────────────────────
    log.debug(
        "stage_route_failed",
        sample_id=sample.sample_id,
        score=score,
        action=action,
    )
    sample.training_stage = ""
    if not sample.quality or not sample.quality.approved_for_training:
        sample.status = SampleStatus.REJECTED.value

    return ""
