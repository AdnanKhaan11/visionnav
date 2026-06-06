"""
Normalizer — converts raw recording dicts to PipelineSample objects.

This is the boundary between "raw data as recorded" and
"data as the pipeline understands it."

Every source format (ActionRecorder, VASH, WikiHow) has its
own normalizer that converts to the canonical PipelineSample.
The rest of the pipeline only ever sees PipelineSample.

This follows the Adapter pattern:
  Raw format A → normalizer_a → PipelineSample
  Raw format B → normalizer_b → PipelineSample
  Raw format C → normalizer_c → PipelineSample
                                      ↓
                               same pipeline for all
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from data_pipeline.core.exceptions import IngestionError
from data_pipeline.core.metadata import now_iso, new_sample_id
from data_pipeline.core.schemas import (
    PipelineSample,
    QualityMetrics,
    RawContent,
    EnrichedContent,
    SampleStatus,
    SourceType,
)


def normalize_recorded_step(
    data: dict,
    source_file: str,
    line_number: int,
    source_type: str = SourceType.HUMAN_DEMO.value,
) -> PipelineSample:
    """
    Convert one RecordedStep dict (from ActionRecorder) to PipelineSample.

    This is the most important normalizer — ActionRecorder is our
    primary data collection tool right now.

    Args:
        data:        raw dict from JSONL (matches RecordedStep fields)
        source_file: path to the JSONL file this came from
        line_number: which line in that file
        source_type: SourceType enum value

    Returns:
        PipelineSample ready for validation

    Raises:
        IngestionError: if required fields are missing or malformed
    """
    try:
        # ── Extract required fields ───────────────────────────────────────
        step_index = int(data["step_index"])
        task = str(data.get("task", "")).strip()
        session_id = str(data.get("session_id", "unknown"))

        if not task:
            raise IngestionError(
                f"Missing 'task' field in line {line_number} of {source_file}"
            )

        # ── Extract action dict ───────────────────────────────────────────
        action = data.get("action", {})
        if not isinstance(action, dict):
            raise IngestionError(
                f"'action' must be a dict, got {type(action).__name__} "
                f"at line {line_number}"
            )

        action_type = str(action.get("type", "")).strip().lower()
        coordinates = action.get("coordinates")
        text = action.get("text")
        key = action.get("key")
        description = str(action.get("description", "")).strip()

        if not action_type:
            raise IngestionError(
                f"Missing action.type at line {line_number} of {source_file}"
            )

        # Normalize coordinates to list[float] or None
        if coordinates is not None:
            try:
                coordinates = [float(coordinates[0]), float(coordinates[1])]
            except (TypeError, IndexError, ValueError):
                coordinates = None  # let coordinate_validator catch this

        # ── Build RawContent ──────────────────────────────────────────────
        raw = RawContent(
            screenshot_path=str(data.get("screenshot_path", "")),
            action_type=action_type,
            coordinates=coordinates,
            text=str(text) if text is not None else None,
            key=str(key) if key is not None else None,
            description=description,
            ocr_text_raw=str(data.get("ocr_text", "")),
        )

        # ── Build PipelineSample ──────────────────────────────────────────
        sample = PipelineSample(
            sample_id=new_sample_id(),
            session_id=session_id,
            step_index=step_index,
            source_type=source_type,
            task=task,
            status=SampleStatus.RAW.value,
            created_at=data.get("timestamp", now_iso()),
            updated_at=now_iso(),
            raw=raw,
            enriched=EnrichedContent(),  # empty — enrichment fills this
            quality=QualityMetrics(),  # empty — validation fills this
            tags=_infer_tags(task, action_type),
        )

        return sample

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Unexpected error normalizing line {line_number} "
            f"of {source_file}: {exc}"
        )


def normalize_synthetic_episode_step(
    data: dict,
    source_file: str,
    line_number: int,
) -> PipelineSample:
    """
    Convert one VASH EpisodeStep dict to PipelineSample.

    VASH output format is different from ActionRecorder:
      - No screenshot (simulation has no real screen)
      - Uses state_desc instead of ocr_text
      - action is a string like "key:win+r" not a dict

    Stub for now — implement when VASH export is built.
    """
    raise NotImplementedError(
        "VASH source normalizer not yet implemented. "
        "Build VASH episode export first."
    )


def _infer_tags(task: str, action_type: str) -> list[str]:
    """
    Infer searchable tags from task text and action type.
    Tags enable fast filtering without reading full sample content.
    """
    tags: list[str] = []
    task_lower = task.lower()

    # Task category tags
    category_keywords = {
        "email": ["email", "gmail", "outlook", "inbox", "message"],
        "browser": ["chrome", "browser", "navigate", "search", "url"],
        "file": ["file", "folder", "save", "open", "document"],
        "form": ["form", "fill", "input", "submit", "register"],
        "android": ["android", "phone", "mobile", "app"],
        "multilingual": ["urdu", "pashto", "arabic", "hindi"],
    }

    for tag, keywords in category_keywords.items():
        if any(kw in task_lower for kw in keywords):
            tags.append(tag)

    # Action type tag
    tags.append(f"action:{action_type}")

    return tags
