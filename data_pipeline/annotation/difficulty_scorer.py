"""
Difficulty scorer — assigns 1-5 difficulty rating to each sample.

Two modes:
  1. Rule-based (fast, free, ~80% accuracy)
  2. LLM-based (slower, costs money, ~95% accuracy)

For Stage 1: use rule-based
For Stage 3+: use LLM for hard cases, rule-based for easy cases
"""

from __future__ import annotations

import structlog

from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)

# Task keywords that indicate higher difficulty
_HARD_KEYWORDS = {
    "filter",
    "sort",
    "search",
    "find",
    "compare",
    "multiple",
    "all",
    "every",
    "between",
    "before",
    "after",
    "export",
    "import",
    "configure",
    "settings",
    "advanced",
    "recover",
    "undo",
    "restore",
    "merge",
    "split",
}

_MEDIUM_KEYWORDS = {
    "reply",
    "forward",
    "attach",
    "compose",
    "create",
    "save",
    "open",
    "close",
    "navigate",
    "select",
    "download",
    "upload",
    "copy",
    "paste",
    "move",
}

# Action types with inherent complexity
_ACTION_COMPLEXITY = {
    "click": 1,
    "double_click": 1,
    "right_click": 2,
    "type": 2,
    "key": 2,
    "scroll": 1,
    "wait": 1,
    "done": 0,
    "fail": 0,
}


def score(sample: PipelineSample) -> int:
    """
    Compute difficulty score using rule-based heuristics.

    Rules are cumulative — each matching rule adds to the score.
    Final score is clamped to [1, 5].

    Args:
        sample: annotated PipelineSample

    Returns:
        int in range [1, 5]
    """
    base_score = 1

    task_lower = sample.task.lower()

    # ── Task complexity keywords ───────────────────────────────────────────
    hard_matches = sum(1 for kw in _HARD_KEYWORDS if kw in task_lower)
    medium_matches = sum(1 for kw in _MEDIUM_KEYWORDS if kw in task_lower)

    if hard_matches >= 3:
        base_score += 3
    elif hard_matches >= 1:
        base_score += 2
    elif medium_matches >= 2:
        base_score += 1

    # ── Step position within trajectory ───────────────────────────────────
    # Later steps in long trajectories are harder
    # (more context to track, more state to maintain)
    if sample.step_index >= 15:
        base_score += 2
    elif sample.step_index >= 8:
        base_score += 1

    # ── Screen complexity ──────────────────────────────────────────────────
    # Dense screens with many elements are harder to navigate
    n_regions = 0
    if sample.enriched:
        n_regions = sample.enriched.n_ocr_regions

    if n_regions >= 50:
        base_score += 2
    elif n_regions >= 25:
        base_score += 1

    # ── Action type complexity ─────────────────────────────────────────────
    if sample.raw:
        action_complexity = _ACTION_COMPLEXITY.get(sample.raw.action_type, 1)
        base_score += action_complexity - 1  # click=0, type=1, right_click=1

    # ── Multilingual penalty (harder for current model) ───────────────────
    if sample.enriched:
        lang = sample.enriched.detected_language
        if lang in ("ur", "ps", "ar"):
            base_score += 1  # multilingual tasks are genuinely harder

    # ── Clamp to [1, 5] ───────────────────────────────────────────────────
    final = max(1, min(5, base_score))

    # Store in annotation if it exists
    if sample.annotation:
        sample.annotation.difficulty = final

    log.debug(
        "difficulty_scored",
        sample_id=sample.sample_id,
        difficulty=final,
        step_index=sample.step_index,
        n_regions=n_regions,
    )

    return final
