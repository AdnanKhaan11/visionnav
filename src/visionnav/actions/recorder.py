"""
ActionRecorder — captures real agent trajectories for training data.

This is Step 1 of the Dataset Factory pipeline:
  Record → Clean → Annotate → Validate → Train

Every training sample in VisionNav starts as a recorded trajectory.
High-quality recordings produce high-quality models.
Low-quality recordings produce confused models.

Design goals:
  - Zero data loss (screenshot + action saved together atomically)
  - Self-contained records (each step makes sense on its own)
  - Resumable sessions (append mode lets you add to existing files)
  - Portable format (JSONL — one JSON object per line, easy to stream)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import structlog

from visionnav.actions.schema import Action
from visionnav.utils.image import save_screenshot

log = structlog.get_logger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class RecordedStep:
    """
    One step in a recorded trajectory.

    Self-contained means: you can read this step alone and understand
    everything about it — no need to read the other steps first.

    This is important because:
      - Steps may be loaded individually (streaming)
      - Steps may be shuffled during training
      - Steps may be reused across different training stages

    After annotation, this becomes one training sample.
    """

    step_index: int
    task: str
    action: dict  # Action serialized to plain dict
    screenshot_path: str  # absolute path to the PNG file
    ocr_text: str  # what text was on screen
    screen_width: int
    screen_height: int
    timestamp: str  # ISO 8601 format
    reasoning: str = ""  # filled later by annotation pipeline
    verified: bool = False  # True after human review
    session_id: str = ""


@dataclass
class RecordingSession:
    """
    All metadata for one complete recording session.

    A session = one human or agent attempting one task.
    Contains: metadata about the task + all steps recorded.
    """

    session_id: str
    task: str
    platform: str  # "windows" | "macos" | "android"
    started_at: str  # ISO 8601
    finished_at: str = ""
    success: bool = False
    total_steps: int = 0
    screenshot_dir: str = ""
    steps: list[RecordedStep] = field(default_factory=list)


# ─── Recorder ─────────────────────────────────────────────────────────────────


class ActionRecorder:
    """
    Records agent trajectories to JSONL files for training data collection.

    Two ways to use it:

    Way 1 — Context manager (recommended):
        with ActionRecorder(task="Open Gmail", output_dir=Path("data")) as r:
            r.record_step(action, screenshot, ocr_text, meta)
            r.record_step(action, screenshot, ocr_text, meta)
        # Session automatically saved on exit

    Way 2 — Manual:
        r = ActionRecorder(task="Open Gmail", output_dir=Path("data"))
        r.start()
        r.record_step(...)
        r.finish(success=True)
    """

    def __init__(
        self,
        task: str,
        output_dir: Path,
        platform: str = "windows",
        session_id: str | None = None,
    ) -> None:
        self._task = task
        self._output_dir = Path(output_dir)
        self._platform = platform

        # Generate short unique ID if none provided
        # We use only first 8 characters for readability
        self._session_id = session_id or str(uuid.uuid4())[:8]

        # Screenshots for this session go in their own subdirectory
        # Keeps things organized: screenshots/session_abc123/step_001.png
        self._screenshot_dir = self._output_dir / "screenshots" / self._session_id

        # Create directories now — fail early if permissions are wrong
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session: RecordingSession | None = None
        self._step_count = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def output_path(self) -> Path:
        """Path to the JSONL file where steps are saved."""
        return self._output_dir / f"session_{self._session_id}.jsonl"

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Begin a new recording session.

        Creates the RecordingSession object and logs the start.
        Must be called before record_step().
        """
        self._session = RecordingSession(
            session_id=self._session_id,
            task=self._task,
            platform=self._platform,
            started_at=datetime.now(timezone.utc).isoformat(),
            screenshot_dir=str(self._screenshot_dir),
        )

        log.info(
            "recording_started",
            session_id=self._session_id,
            task=self._task,
            output=str(self.output_path),
        )

    def record_step(
        self,
        action: Action,
        screenshot: np.ndarray,
        ocr_text: str,
        screen_meta: dict,
    ) -> RecordedStep:
        """
        Record one step: save screenshot to disk and build RecordedStep.

        Call this AFTER the action executes — screenshot should show
        the screen state AFTER the action completed.

        Args:
            action:      the Action that was executed
            screenshot:  numpy array (H, W, 3) uint8
            ocr_text:    text from OCR (joined into one string)
            screen_meta: {"width": int, "height": int, ...}

        Returns:
            RecordedStep with all fields filled

        Raises:
            RuntimeError: if start() was not called first
        """
        if self._session is None:
            raise RuntimeError(
                "Recording session not started. "
                "Call start() or use the context manager."
            )

        step_index = self._step_count

        # ── Save screenshot to disk ──────────────────────────────────────
        # Filename: step_001.png, step_002.png, etc.
        # Zero-padded to 3 digits so files sort correctly in file explorer
        screenshot_filename = f"step_{step_index:03d}.png"
        screenshot_path = self._screenshot_dir / screenshot_filename
        save_screenshot(screenshot, screenshot_path)

        # ── Serialize action to plain dict ───────────────────────────────
        # We store as dict (not Action object) so JSONL can serialize it
        # action.model_dump() converts Pydantic model → plain Python dict
        action_dict = action.model_dump()

        # Convert any enum values to their string representation
        # Because JSON does not know about Python enums
        if "type" in action_dict:
            action_dict["type"] = str(
                action_dict["type"].value
                if hasattr(action_dict["type"], "value")
                else action_dict["type"]
            )

        # ── Build the RecordedStep ────────────────────────────────────────
        step = RecordedStep(
            step_index=step_index,
            task=self._task,
            action=action_dict,
            screenshot_path=str(screenshot_path.resolve()),
            ocr_text=ocr_text,
            screen_width=screen_meta.get("width", screenshot.shape[1]),
            screen_height=screen_meta.get("height", screenshot.shape[0]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self._session_id,
        )

        # ── Append to session and increment counter ───────────────────────
        self._session.steps.append(step)
        self._step_count += 1

        log.debug(
            "step_recorded",
            step_index=step_index,
            action_type=action_dict.get("type", "unknown"),
            session_id=self._session_id,
        )

        return step

    def finish(self, success: bool = True) -> RecordingSession:
        """
        End the session and write all steps to the JSONL file.

        Format: one JSON object per line.
        Why one line per step (not one big JSON array)?
          - Can stream large files line by line (no loading into RAM)
          - Can append more steps later (just add more lines)
          - Partial files are still readable (not broken like JSON arrays)

        Args:
            success: did the task complete successfully?

        Returns:
            The completed RecordingSession
        """
        if self._session is None:
            raise RuntimeError("No active session to finish.")

        # ── Update session metadata ───────────────────────────────────────
        self._session.finished_at = datetime.now(timezone.utc).isoformat()
        self._session.success = success
        self._session.total_steps = self._step_count

        # ── Write each step as one JSON line ─────────────────────────────
        # mode="a" = append — safe to call finish() multiple times
        # Each line is complete JSON that can be parsed independently
        with open(self.output_path, "a", encoding="utf-8") as f:
            for step in self._session.steps:
                line = json.dumps(asdict(step), ensure_ascii=False)
                f.write(line + "\n")

        log.info(
            "recording_finished",
            session_id=self._session_id,
            total_steps=self._step_count,
            success=success,
            output=str(self.output_path),
        )

        return self._session

    def load(self, jsonl_path: Path) -> Iterator[RecordedStep]:
        """
        Stream RecordedStep objects from a JSONL file.

        Uses yield (generator) — reads one line at a time.
        Never loads the entire file into memory.
        This is critical for large files (10,000+ steps).

        Usage:
            for step in recorder.load(Path("data/session_abc.jsonl")):
                process_step(step)

        Yields:
            RecordedStep objects in order (skips empty/corrupt lines)
        """
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):

                # Skip empty lines (common at end of file)
                line = line.strip()
                if not line:
                    continue

                # Parse JSON — skip lines that are corrupt
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning(
                        "jsonl_parse_error",
                        file=str(jsonl_path),
                        line=line_number,
                        error=str(exc),
                    )
                    continue

                # Convert dict back to RecordedStep
                # We only pass fields that RecordedStep knows about
                # Extra fields in the JSON are silently ignored
                try:
                    step = RecordedStep(
                        step_index=data["step_index"],
                        task=data["task"],
                        action=data["action"],
                        screenshot_path=data["screenshot_path"],
                        ocr_text=data["ocr_text"],
                        screen_width=data["screen_width"],
                        screen_height=data["screen_height"],
                        timestamp=data["timestamp"],
                        reasoning=data.get("reasoning", ""),
                        verified=data.get("verified", False),
                        session_id=data.get("session_id", ""),
                    )
                    yield step

                except (KeyError, TypeError) as exc:
                    log.warning(
                        "jsonl_missing_field",
                        file=str(jsonl_path),
                        line=line_number,
                        error=str(exc),
                    )
                    continue

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ActionRecorder":
        """Called when entering 'with ActionRecorder(...) as r:'"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Called when exiting the 'with' block.
        success=True if no exception was raised inside the block.
        success=False if an exception occurred.
        """
        success = exc_type is None
        self.finish(success=success)
        # Return None (not True) so exceptions propagate normally
