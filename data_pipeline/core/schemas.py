"""
Core data schemas for the Dataset Factory pipeline.

Every stage of the pipeline works with PipelineSample.
This is the internal canonical format — not the raw recording format,
not the training output format, but the pipeline's own representation.

The pipeline transforms data through these stages:
  RecordedStep (raw)
    → PipelineSample (pipeline internal)
      → [validate → enrich → annotate → curate]
        → TrainingExample (LLaMA-Factory output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ─── Enums ────────────────────────────────────────────────────────────────────


class SampleStatus(Enum):
    """
    Lifecycle state of a sample through the pipeline.
    A sample moves forward through these states.
    It never moves backward (except quarantine → review → active).
    """

    RAW = "raw"  # just ingested, nothing done yet
    VALIDATING = "validating"  # currently being validated
    QUARANTINED = "quarantined"  # failed a gate, needs human review
    ENRICHING = "enriching"  # validation passed, enrichment running
    ENRICHED = "enriched"  # enrichment complete, ready for annotation
    ANNOTATING = "annotating"  # annotation in progress
    ANNOTATED = "annotated"  # annotation complete, ready for curation
    APPROVED = "approved"  # passed curation, ready for export
    REJECTED = "rejected"  # permanently rejected, will not be trained on
    EXPORTED = "exported"  # included in a training dataset release


class SourceType(Enum):
    """Where did this sample come from?"""

    HUMAN_DEMO = "human_demo"  # human performed task, recorded
    SYNTHETIC = "synthetic"  # VASH simulation
    TUTORIAL = "tutorial"  # crawled from WikiHow/Baidu/YouTube
    SELF_PLAY = "self_play"  # agent attempted task in production
    CORRECTION = "correction"  # human corrected a failed trajectory
    REPLAY = "replay"  # replayed an existing trajectory


class Platform(Enum):
    """What platform was the screenshot taken on?"""

    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    ANDROID = "android"
    WEB = "web"
    UNKNOWN = "unknown"


class Language(Enum):
    """Detected primary language of the UI."""

    ENGLISH = "en"
    URDU = "ur"
    PASHTO = "ps"
    ARABIC = "ar"
    HINDI = "hi"
    CHINESE = "zh"
    UNKNOWN = "unknown"


class TrainingStage(Enum):
    """Which LLaMA-Factory training stage this sample belongs to."""

    STAGE1_GROUNDING = "stage1_grounding"  # element location
    STAGE2_ACTION = "stage2_action"  # action prediction
    STAGE3_PLANNING = "stage3_planning"  # multi-step reasoning
    ALL_STAGES = "all_stages"  # high quality → all three


# ─── Core Sample Schema ───────────────────────────────────────────────────────


@dataclass
class RawContent:
    """
    The original recorded content — never modified after ingestion.
    Immutable record of what actually happened during recording.
    """

    screenshot_path: str
    action_type: str  # from ActionType enum value
    coordinates: list[float] | None
    text: str | None
    key: str | None
    description: str
    ocr_text_raw: str  # OCR text captured at recording time


@dataclass
class EnrichedContent:
    """
    Content added by the enrichment layer.
    More accurate/complete than what was captured at recording time.
    """

    ocr_regions_detailed: list[dict] = field(default_factory=list)
    # Each dict: {"text": str, "bbox": [x1,y1,x2,y2], "confidence": float}

    image_hash_sha256: str = ""  # exact duplicate detection
    image_hash_phash: str = ""  # near-duplicate detection

    detected_language: str = Language.UNKNOWN.value
    detected_platform: str = Platform.UNKNOWN.value

    detected_app: str = ""  # "chrome", "gmail", "notepad", etc.
    n_ocr_regions: int = 0
    n_ui_elements: int = 0

    image_width: int = 0
    image_height: int = 0
    image_file_size_kb: float = 0.0


@dataclass
class Annotation:
    """
    Semantic meaning added by the annotation layer.
    Either auto-generated (by LLM) or human-provided.
    """

    reasoning: str = ""  # why was this action taken?
    intent: str = ""  # what goal does this serve?
    difficulty: int = 0  # 1=trivial, 5=expert-level
    is_error_step: bool = False  # True if this step corrects a mistake
    error_type: str = ""  # "wrong_click", "missed_element", etc.

    # Annotation provenance
    annotated_by: str = ""  # "claude-sonnet-4-6", "human_pk_001"
    annotation_method: str = ""  # "auto_llm", "human", "human_verified"
    annotation_confidence: float = 0.0  # 0.0 = uncertain, 1.0 = certain

    # Cross-reference validation result
    reasoning_verified: bool = False  # reasoning elements found in OCR
    verified_at: str = ""  # ISO timestamp


@dataclass
class QualityMetrics:
    """
    Quality assessment computed by the curation layer.
    Composite score determines final approval decision.
    """

    # Individual gate results
    schema_valid: bool = False
    image_valid: bool = False
    coordinates_valid: bool = True  # True if no coordinates (non-click)
    no_pii_detected: bool = True
    not_duplicate: bool = True

    # Trajectory-level quality (from TrajectoryAnalyzer)
    trajectory_quality: float = 0.0  # from TrajectoryAnalyzer.analyze()
    loop_detected: bool = False
    redundant_actions: bool = False

    # Annotation quality
    has_reasoning: bool = False
    reasoning_length: int = 0
    reasoning_verified: bool = False

    # Composite score
    quality_score: float = 0.0  # 0.0 to 1.0
    approved_for_training: bool = False

    # Rejection reason (if rejected)
    rejection_reason: str = ""


@dataclass
class PipelineSample:
    """
    The canonical internal representation of one training sample.

    This is the object that flows through every pipeline stage.
    Every module receives a PipelineSample and returns a PipelineSample.

    Designed for:
      - Forward compatibility (add fields without breaking old code)
      - Complete auditability (every change is traceable)
      - Efficient serialization (all fields JSON-compatible)

    One PipelineSample = one screenshot + one action + one reasoning.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    sample_id: str  # globally unique: "vn_{8char_uuid}"
    session_id: str  # which recording session this came from
    step_index: int  # position within the session

    # ── Source ────────────────────────────────────────────────────────────
    source_type: str  # SourceType enum value
    task: str  # the instruction: "Open Gmail and find emails"
    task_category: str = ""  # "email", "browser", "file_mgmt", etc.

    # ── Lifecycle ─────────────────────────────────────────────────────────
    status: str = SampleStatus.RAW.value
    created_at: str = ""
    updated_at: str = ""

    # ── Content layers ─────────────────────────────────────────────────────
    raw: RawContent | None = None
    enriched: EnrichedContent | None = None
    annotation: Annotation | None = None
    quality: QualityMetrics | None = None
    rejection_reason: str | None = None

    # ── Training assignment ────────────────────────────────────────────────
    training_stage: str = ""  # TrainingStage enum value
    dataset_version: str = ""  # "2.1.0" — which release included this

    # ── Schema version ────────────────────────────────────────────────────
    schema_version: str = "1.0.0"
    # When schema changes, increment this.
    # Old samples with lower version need migration before use.

    # ── Extra fields ──────────────────────────────────────────────────────
    tags: list[str] = field(default_factory=list)
    # Free-form tags: ["multilingual", "error_recovery", "long_horizon"]
    # Used for filtering and analysis.

    extra: dict[str, Any] = field(default_factory=dict)
    # Forward-compatible escape hatch.
    # Never put required fields here — use named fields above.

    # ── Helpers ───────────────────────────────────────────────────────────

    def is_approved(self) -> bool:
        return self.status == SampleStatus.APPROVED.value

    def is_rejected(self) -> bool:
        return self.status == SampleStatus.REJECTED.value

    def is_multilingual(self) -> bool:
        if self.enriched is None:
            return False
        return self.enriched.detected_language not in (
            Language.ENGLISH.value,
            Language.UNKNOWN.value,
        )

    def quality_score(self) -> float:
        if self.quality is None:
            return 0.0
        return self.quality.quality_score

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSONL storage."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineSample":
        """Reconstruct from dict (JSONL load)."""
        # Reconstruct nested dataclasses from dicts
        sample = cls(
            sample_id=data["sample_id"],
            session_id=data["session_id"],
            step_index=data["step_index"],
            source_type=data["source_type"],
            task=data["task"],
            task_category=data.get("task_category", ""),
            status=data.get("status", SampleStatus.RAW.value),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            training_stage=data.get("training_stage", ""),
            dataset_version=data.get("dataset_version", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            tags=data.get("tags", []),
            extra=data.get("extra", {}),
        )

        if data.get("raw"):
            sample.raw = RawContent(**data["raw"])

        if data.get("enriched"):
            sample.enriched = EnrichedContent(**data["enriched"])

        if data.get("annotation"):
            sample.annotation = Annotation(**data["annotation"])

        if data.get("quality"):
            sample.quality = QualityMetrics(**data["quality"])

        return sample
