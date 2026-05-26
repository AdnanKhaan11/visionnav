You are right. I skipped it. My fault. Let me fix that now.

---

# SESSION 3B — The Action System: The Hands of VisionNav

The action system answers one question:

> **How does the agent's decision become a real effect on the computer?**

The answer involves four files working as a pipeline:

```
Model text output
      ↓
parser.py      → convert text → typed Action object
      ↓
safety/classifier.py → is this safe to do?
      ↓
executor.py    → route Action to the right OS call
      ↓
platforms/desktop.py → pyautogui/ADB actually moves mouse
      ↓
verifier.py    → did the screen change as expected?
```

Each file has exactly one job. Let us go deep into each.

---

## Deep Dive 1 — schema.py: The Action Vocabulary

The schema defines what actions are POSSIBLE. Nothing more.

```python
class ActionType(str, Enum):
    CLICK        = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK  = "right_click"
    TYPE         = "type"
    KEY          = "key"
    SCROLL       = "scroll"
    WAIT         = "wait"
    DONE         = "done"
    FAIL         = "fail"
```

**Why `str, Enum` and not just `Enum`?**

```python
# Regular Enum
class Color(Enum):
    RED = "red"

Color.RED == "red"        # False  ← painful
json.dumps(Color.RED)     # TypeError ← crashes
str(Color.RED)            # "Color.RED" ← ugly

# str, Enum (our approach)
class ActionType(str, Enum):
    CLICK = "click"

ActionType.CLICK == "click"       # True  ← clean
json.dumps(ActionType.CLICK)      # '"click"' ← works
ActionType("click") == ActionType.CLICK  # True ← deserializes
```

`str, Enum` makes the enum behave like a string everywhere — in JSON,
in SQLite, in API responses — while still being a validated typed object.
This is why we can store `action.type` directly in the database as a string
and reconstruct it with `ActionType(stored_string)`.

**The Action dataclass:**

```python
class Action(BaseModel):
    type:        ActionType
    coordinates: tuple[float, float] | None = None
    text:        str | None = None
    key:         str | None = None
    direction:   str | None = None
    amount:      int | None = None
    description: str = ""
    confidence:  float = Field(default=1.0, ge=0.0, le=1.0)
```

Why Pydantic `BaseModel` instead of `@dataclass`?

```
@dataclass:
  Simple container. No validation. No serialization help.
  Action(type="invalid_type") → no error, silent bug

Pydantic BaseModel:
  Validates on creation. Serializes to JSON. Type coerces.
  Action(type="invalid_type") → ValidationError immediately
  Action(type="click") → ActionType.CLICK automatically coerced
  action.model_dump() → clean dict
  Action.model_validate(dict) → clean reconstruction
```

Every invalid action fails at the boundary (schema creation)
not deep inside the system where it is hard to debug.

---

## Deep Dive 2 — parser.py: The Most Defensive Code We Write

The model outputs raw text. That text can be:
```
Perfect:   <think>I see a button</think><action>{"type":"click",...}</action>
Partial:   <action>{"type":"click","coordinates":[0.5,}  ← malformed JSON
Empty:     "I'll help you with that!"    ← no action block at all
Confused:  <action>click the button</action>  ← not JSON
Extra:     Here is my action: <action>{"type":"done"}</action> Extra text
```

The parser must handle ALL of these without crashing.

```python
_PATTERN = re.compile(
    r"<action>\s*(.*?)\s*</action>",
    re.DOTALL | re.IGNORECASE
)

def parse_action(raw: str) -> Action:

    # Step 1: Find <action> block
    match = _PATTERN.search(raw)
    if not match:
        raise ActionParseError(
            f"No <action> block found. Preview: {raw[:80]!r}"
        )

    content = match.group(1).strip()

    # Step 2: Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON inside <action>: {exc}")

    # Step 3: Validate action type
    raw_type = data.get("type", "")
    try:
        action_type = ActionType(raw_type)
    except ValueError:
        raise ActionParseError(
            f"Unknown action type: {raw_type!r}. "
            f"Valid types: {[t.value for t in ActionType]}"
        )

    # Step 4: Validate and normalize coordinates
    coords = data.get("coordinates")
    if coords is not None:
        if not isinstance(coords, (list, tuple)) or len(coords) != 2:
            raise ActionParseError("'coordinates' must be [x, y]")
        nx, ny = float(coords[0]), float(coords[1])
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise ActionParseError(
                f"Coordinates [{nx}, {ny}] out of [0,1] range"
            )
        coords = (round(nx, 4), round(ny, 4))

    return Action(
        type        = action_type,
        coordinates = coords,
        text        = data.get("text"),
        key         = data.get("key"),
        direction   = data.get("direction"),
        amount      = data.get("amount"),
        description = data.get("description", ""),
    )
```

**Why is every failure a specific `ActionParseError` and not a generic `Exception`?**

Because the agent loop catches `ActionParseError` specifically:

```python
try:
    action = parse_action(raw_output)
except ActionParseError as exc:
    # Known failure → handle gracefully
    action = Action(type=ActionType.FAIL, description=str(exc))
```

If we raised a generic `Exception`, the agent might catch too much —
or miss it entirely. Specific exceptions are contracts:
"This function fails in exactly these known ways."

---

## Deep Dive 3 — executor.py: The Dispatcher Pattern

The executor translates an `Action` object into OS operations.

```python
async def execute(
    self,
    action: Action,
    screen_w: int,
    screen_h: int,
) -> bool:

    # DONE and FAIL actions don't need OS execution
    if action.type in (ActionType.DONE, ActionType.FAIL, ActionType.WAIT):
        return True

    return await self._dispatch(action, screen_w, screen_h)


async def _dispatch(
    self,
    action: Action,
    w: int,
    h: int,
) -> bool:
    match action.type:
        case ActionType.CLICK | ActionType.DOUBLE_CLICK | ActionType.RIGHT_CLICK:
            if action.coordinates is None:
                return False
            x, y = self._to_pixels(action.coordinates, w, h)
            click_type = {
                ActionType.CLICK:        "left",
                ActionType.DOUBLE_CLICK: "double",
                ActionType.RIGHT_CLICK:  "right",
            }[action.type]
            return await self._platform.execute_click(x, y, click_type)

        case ActionType.TYPE:
            return await self._platform.execute_type(action.text or "")

        case ActionType.KEY:
            return await self._platform.execute_key(action.key or "")

        case ActionType.SCROLL:
            if action.coordinates:
                x, y = self._to_pixels(action.coordinates, w, h)
            else:
                x, y = w // 2, h // 2   # default: scroll at center
            return await self._platform.execute_scroll(
                x, y, action.direction or "down", action.amount or 3
            )

        case _:
            log.warning("unknown_action_type", action_type=action.type)
            return False


def _to_pixels(
    self,
    normalized: tuple[float, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    """Convert normalized [0,1] coordinates to pixel coordinates."""
    nx, ny = normalized
    return round(nx * width), round(ny * height)
```

**The `match` statement (Python 3.10+) is perfect here because:**

```
Each case is a complete branch: exactly one action type, exactly one behavior.
The `case _:` catch-all logs unknown types instead of crashing.
Adding a new action type = adding one `case` block. Nothing else changes.
```

This is the **dispatcher pattern** — a central router that delegates to
specialized handlers. The alternative (a chain of `if/elif`) is harder to
read and harder to extend.

---

## Deep Dive 4 — verifier.py: Computer Vision for Verification

```python
class ActionVerifier:

    def __init__(self, threshold: float = 0.01) -> None:
        self._threshold = threshold

    def verify(
        self,
        before: np.ndarray,
        after:  np.ndarray,
        action: Action,
    ) -> tuple[bool, float]:

        # These actions don't change screen — skip verification
        if action.type in _NO_VERIFY_TYPES:
            return True, 0.0

        # Compute pixel-level difference
        diff         = np.abs(
            before.astype(np.int16) - after.astype(np.int16)
        )
        change_ratio = float((diff > 30).mean())

        return change_ratio >= self._threshold, change_ratio
```

**Why `int16` not `int32` or `float32`?**

```python
before = np.array([200, 200, 200], dtype=np.uint8)
after  = np.array([50,  50,  50],  dtype=np.uint8)

# Wrong: uint8 subtraction wraps around
np.abs(before.astype(np.uint8) - after.astype(np.uint8))
# 200 - 50 = 150... fine
# 50 - 200 = ??? In uint8: wraps to 106 (not 150!)
# np.abs does not fix this!

# Correct: int16 can hold -255 to 255
np.abs(before.astype(np.int16) - after.astype(np.int16))
# 200 - 50 = 150 ✓
# 50 - 200 = -150, abs = 150 ✓
```

`uint8` arithmetic wraps around. `int16` does not.
This is a subtle bug that would make your verifier wrong 50% of the time
on dark→light screen changes.

**Why `> 30` threshold for "changed"?**

```
LCD screens have pixel noise: ±5 intensity units constantly
Anti-aliasing changes pixels: ±10-15 units on text edges
Clock updates change pixels: ~20 pixels in the corner
Real UI change: hundreds to thousands of pixels change by 50-200 units

Threshold of 30 means:
  Normal noise (< 30) → ignored (not "changed")
  Real UI element change (> 30) → counted as "changed"
```

The threshold separates signal from noise.

---

## Deep Dive 5 — How Safety Integrates With the Action Pipeline

```python
# In safety/classifier.py
RISK_TABLE: dict[ActionType, RiskLevel] = {
    ActionType.CLICK:        RiskLevel.LOW,
    ActionType.DOUBLE_CLICK: RiskLevel.LOW,
    ActionType.TYPE:         RiskLevel.MEDIUM,
    ActionType.KEY:          RiskLevel.MEDIUM,
    ActionType.SCROLL:       RiskLevel.SAFE,
    ActionType.WAIT:         RiskLevel.SAFE,
    ActionType.DONE:         RiskLevel.SAFE,
    ActionType.FAIL:         RiskLevel.SAFE,
}

DANGEROUS_KEYS = {"delete", "f4", "ctrl+alt+del", ...}
```

Safety sits between parsing and execution:

```
parse_action() → Action
                   ↓
safety.classify(action)
                   ↓
        ┌──────────┴──────────┐
      SAFE/LOW/MEDIUM        HIGH/BLOCKED
        ↓                       ↓
   executor.execute()     create FAIL action
                          do NOT execute
```

The safety classifier is intentionally simple and fast.
It does not need a model — it is pure rule-based classification.
Speed matters here: every step goes through it.

---

---

# ASSIGNMENT 5 — The Action Intelligence System

## What You Are Building

A production-grade **ActionRecorder** and **TrajectoryAnalyzer** system.

This is not just an exercise — it is the foundation of Phase 16 (Dataset Factory).
Every training sample we create starts with recorded actions.
Every recorded trajectory needs quality analysis before becoming training data.

This assignment trains:
- Deep understanding of the action schema (you work with it directly)
- Serialization and deserialization (save/load to JSON files)
- Pattern detection (algorithmic thinking)
- Data pipeline design (real engineering)
- File I/O and path handling
- Clean class design with single responsibilities

---

## Part A — ActionRecorder

Create `src/visionnav/actions/recorder.py`:

```python
"""
ActionRecorder — captures real agent trajectories for training data.

This is the first step of the Dataset Factory pipeline:
  Record → Clean → Annotate → Validate → Train

Every training sample in our dataset starts as a recorded trajectory.
High-quality recordings = high-quality model.

Design goals:
  - Zero data loss (every action + screenshot saved atomically)
  - Self-contained records (each step has everything needed)
  - Resumable sessions (can add steps to existing recording)
  - Portable format (JSONL — one JSON object per line)
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

from visionnav.actions.schema import Action, ActionType
from visionnav.utils.image import save_screenshot

log = structlog.get_logger(__name__)


@dataclass
class RecordedStep:
    """
    One recorded step in a trajectory.
    Self-contained: you can understand this step without
    reading any other step in the trajectory.

    This becomes one training sample after annotation.
    """
    step_index:    int
    task:          str
    action:        dict          # Action serialized to dict
    screenshot_path: str         # absolute path to PNG
    ocr_text:      str           # what OCR saw
    screen_width:  int
    screen_height: int
    timestamp:     str           # ISO 8601
    reasoning:     str = ""      # filled during annotation phase
    verified:      bool = False  # True after human review
    session_id:    str = ""


@dataclass
class RecordingSession:
    """
    A complete recording session — one task attempt.
    Contains all steps + metadata.
    """
    session_id:   str
    task:         str
    platform:     str              # "windows" | "macos" | "android"
    started_at:   str
    finished_at:  str = ""
    success:      bool = False
    total_steps:  int = 0
    screenshot_dir: str = ""
    steps:        list[RecordedStep] = field(default_factory=list)


class ActionRecorder:
    """
    Records real agent trajectories as they execute.

    Usage:
        recorder = ActionRecorder(
            task="Open Gmail and find unread emails",
            output_dir=Path("data/recordings"),
        )

        with recorder.session() as session:
            for step in agent_steps:
                session.record(action, screenshot, ocr_text, meta)

        # After context exits: everything saved to JSONL
        print(f"Saved to {recorder.output_path}")
    """

    def __init__(
        self,
        task:           str,
        output_dir:     Path,
        platform:       str = "windows",
        session_id:     str | None = None,
    ) -> None:
        self._task       = task
        self._output_dir = Path(output_dir)
        self._platform   = platform
        self._session_id = session_id or str(uuid.uuid4())[:8]

        # Directory for screenshots of this session
        self._screenshot_dir = (
            self._output_dir / "screenshots" / self._session_id
        )
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._session: RecordingSession | None = None
        self._step_count = 0

    @property
    def output_path(self) -> Path:
        """Path where the JSONL recording will be saved."""
        return self._output_dir / f"session_{self._session_id}.jsonl"

    @property
    def session_id(self) -> str:
        return self._session_id

    def start(self) -> None:
        """
        Begin a new recording session.
        Creates the RecordingSession object.
        """
        # Your implementation here
        # Create RecordingSession with:
        #   session_id, task, platform, started_at (ISO timestamp)
        # Set self._session
        # Log: "recording_started" with task and session_id

    def record_step(
        self,
        action:        Action,
        screenshot:    np.ndarray,
        ocr_text:      str,
        screen_meta:   dict,
    ) -> RecordedStep:
        """
        Record one step: save screenshot + serialize action + build RecordedStep.

        This method is called AFTER an action executes.
        The screenshot is the AFTER screenshot (what happened).

        Args:
            action:      the Action that was executed
            screenshot:  numpy array (H, W, 3) — screen after action
            ocr_text:    text extracted from screen by OCR
            screen_meta: {"width": int, "height": int, ...}

        Returns:
            RecordedStep with all fields filled

        Raises:
            RuntimeError: if session not started (call start() first)
        """
        if self._session is None:
            raise RuntimeError(
                "Session not started. Call start() before record_step()."
            )

        # Your implementation here:
        # 1. Save screenshot to self._screenshot_dir / f"step_{step_index:03d}.png"
        #    Use save_screenshot() from utils.image
        # 2. Build RecordedStep with all fields
        # 3. Append to self._session.steps
        # 4. Increment self._step_count
        # 5. Log "step_recorded" with step_index and action type
        # 6. Return the RecordedStep

    def finish(self, success: bool = True) -> RecordingSession:
        """
        End the recording session and save everything to JSONL.

        One JSONL line per step (not the whole session as one object).
        This allows streaming reads of large recording files.

        Format per line:
            {"session_id": "...", "step_index": 0, "task": "...", ...}

        Args:
            success: did the task complete successfully?

        Returns:
            The completed RecordingSession
        """
        if self._session is None:
            raise RuntimeError("No active session to finish.")

        # Your implementation here:
        # 1. Set session.finished_at, session.success, session.total_steps
        # 2. Write each step as one JSON line to self.output_path
        #    (mode="a" to append — allows resuming a recording file)
        # 3. Log "recording_finished" with total_steps and success
        # 4. Return self._session

    def load(self, jsonl_path: Path) -> Iterator[RecordedStep]:
        """
        Stream recorded steps from a JSONL file.

        Uses a generator (yield) — does not load entire file into memory.
        Critical for large recording files (10,000+ steps).

        Usage:
            for step in recorder.load(path):
                process(step)

        Yields:
            RecordedStep objects in order
        """
        # Your implementation here:
        # Open file, iterate lines, parse JSON, yield RecordedStep
        # Skip empty lines and lines that fail to parse (log warning)

    def __enter__(self) -> "ActionRecorder":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        success = exc_type is None    # success if no exception
        self.finish(success=success)
```

---

## Part B — TrajectoryAnalyzer

Create `src/visionnav/actions/trajectory_analyzer.py`:

```python
"""
TrajectoryAnalyzer — quality analysis of recorded trajectories.

Before a recording becomes training data it must pass quality checks.
Bad training data produces bad models.

"Your model quality cannot exceed the quality of your dataset."

This analyzer detects:
  - Action loops (agent stuck repeating the same action)
  - Redundant actions (consecutive identical actions)
  - Low diversity (too many of the same action type)
  - Suspicious coordinates (always clicking same spot = probably wrong)
  - Empty tasks (zero steps recorded)

Each issue gets a severity score. Trajectories below quality_threshold
are flagged for human review.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

from visionnav.actions.recorder import RecordedStep
from visionnav.actions.schema import ActionType


class IssueSeverity(Enum):
    INFO    = "info"      # interesting but not a problem
    WARNING = "warning"   # possible problem, check manually
    ERROR   = "error"     # definitely bad, do not use for training


@dataclass(frozen=True)
class TrajectoryIssue:
    """A detected quality issue in a trajectory."""
    severity:    IssueSeverity
    issue_type:  str         # "loop_detected", "redundant_actions", etc.
    description: str         # human readable: "Steps 3-6 repeat action CLICK at [0.5, 0.5]"
    step_indices: tuple[int, ...] = ()  # which steps are affected


@dataclass
class QualityReport:
    """
    Complete quality analysis of one trajectory.

    quality_score:
        1.0 = perfect
        0.8 = minor issues, probably fine for training
        0.5 = significant issues, human review needed
        0.0 = completely unusable

    use_for_training:
        True if quality_score >= threshold (default 0.6)
    """
    session_id:      str
    task:            str
    total_steps:     int
    action_counts:   dict[str, int]          # {"click": 5, "type": 2, ...}
    issues:          list[TrajectoryIssue]   # all detected issues
    quality_score:   float                   # 0.0 to 1.0
    use_for_training: bool
    summary:         str                     # human-readable verdict

    @property
    def errors(self) -> list[TrajectoryIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[TrajectoryIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]


class TrajectoryAnalyzer:
    """
    Analyzes recorded trajectories for training data quality.

    Usage:
        analyzer = TrajectoryAnalyzer(quality_threshold=0.6)
        report   = analyzer.analyze(steps)
        if report.use_for_training:
            format_as_training_sample(steps)
        else:
            send_for_human_review(steps, report)
    """

    def __init__(
        self,
        quality_threshold:    float = 0.6,
        loop_window:          int   = 4,    # check for loops of up to N actions
        max_same_action_ratio: float = 0.8, # flag if >80% steps are same action
    ) -> None:
        self._threshold      = quality_threshold
        self._loop_window    = loop_window
        self._max_same_ratio = max_same_action_ratio

    def analyze(
        self,
        steps: list[RecordedStep],
        session_id: str = "",
        task: str = "",
    ) -> QualityReport:
        """
        Full quality analysis of a trajectory.

        Runs all detectors and computes a composite quality score.

        Quality score formula:
          start at 1.0
          ERROR   → subtract 0.3 per issue (min 0.0)
          WARNING → subtract 0.1 per issue (min 0.0)
          Clip to [0.0, 1.0]

        Args:
            steps:      list of RecordedStep in order
            session_id: for the report header
            task:       for the report header

        Returns:
            QualityReport with all findings
        """
        # Your implementation here:
        # 1. Handle empty trajectory → return ERROR report immediately
        # 2. Count action types → action_counts dict
        # 3. Run all detectors, collect issues
        # 4. Compute quality_score
        # 5. Build and return QualityReport

    def detect_action_loops(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect when the agent is stuck in a repeating action loop.

        A loop is detected when a sequence of N actions repeats
        consecutively. For example:
          [click, type, click, type, click, type]
           ─────────────  ─────────────
           loop of 2      repeating

        Algorithm:
          For window_size in range(2, loop_window+1):
            Slide a window of size (2 * window_size) across steps
            If first half == second half → loop detected

        Args:
            steps: trajectory steps

        Returns:
            list of TrajectoryIssue (may be empty if no loops)
        """
        # Your implementation here

    def detect_redundant_actions(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect consecutive identical actions.

        Redundant = same action type AND same coordinates (if click).

        Examples:
          DONE, DONE                   → redundant (should never happen)
          click[0.5,0.5], click[0.5,0.5] → redundant (double click unintended?)
          type"hello", type"hello"     → redundant

        Not redundant:
          click[0.5,0.5], click[0.7,0.2] → different coordinates, fine
          key"enter", key"enter"          → may be intentional (two enters)

        Returns:
            list of TrajectoryIssue (WARNING severity)
        """
        # Your implementation here

    def detect_low_diversity(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect trajectories where one action type dominates.

        If >80% of steps are the same action type → WARNING.
        Exception: DONE and FAIL don't count (always appear once at end).

        Example:
          10 steps: [click, click, click, click, click,
                     click, click, click, click, done]
          click ratio = 9/9 = 100% → WARNING: low diversity

        Why this matters:
          A training sample that is 90% clicks teaches the model
          to always click. We want balanced action diversity.

        Returns:
            list of TrajectoryIssue
        """
        # Your implementation here

    def detect_impossible_coordinates(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect click coordinates that are outside [0, 1] range.

        This should never happen (parser validates coordinates)
        but data corruption or direct file editing can cause it.
        These samples CANNOT be used for training.

        Returns:
            list of TrajectoryIssue (ERROR severity — unusable)
        """
        # Your implementation here
        # For each step with action type CLICK/DOUBLE_CLICK/RIGHT_CLICK:
        #   Get coordinates from action dict
        #   If x < 0 or x > 1 or y < 0 or y > 1 → ERROR issue

    def compute_action_statistics(
        self,
        steps: list[RecordedStep],
    ) -> dict[str, int]:
        """
        Count how many times each action type appears.

        Returns:
            {"click": 5, "type": 2, "key": 1, "done": 1}

        Sorted by count descending.
        """
        # Your implementation here

    def format_report(self, report: QualityReport) -> str:
        """
        Format QualityReport as human-readable text for terminal output.

        Output format:
        ══════════════════════════════════════════════
        Quality Report — session_id
        Task: <task>
        ══════════════════════════════════════════════
        Steps:         12
        Quality Score: 0.85 ✓ (use for training)

        Action Distribution:
          click ████████ 8 (67%)
          type  ████     4 (33%)

        Issues Found: 1 warning, 0 errors
          ⚠ WARNING [loop_detected]
            Steps 3-6: action sequence repeats (click, type, click, type)

        Verdict: APPROVED for training
        ══════════════════════════════════════════════

        Returns:
            Formatted string (ready to print)
        """
        # Your implementation here
```

---

## Part C — Batch Analysis Script

Create `scripts/analyze_recordings.py`:

```python
"""
Batch quality analysis of all recordings in a directory.

Usage:
    python scripts/analyze_recordings.py data/recordings/
    python scripts/analyze_recordings.py data/recordings/ --min-score 0.7
    python scripts/analyze_recordings.py data/recordings/ --show-errors-only

Output:
    Prints quality report for each recording
    At the end: summary statistics

    Example:
    ══════════════════════════════════════════════
    Batch Analysis Summary
    ══════════════════════════════════════════════
    Total recordings:      45
    Approved for training: 38 (84%)
    Flagged for review:     7 (16%)
    Average quality score: 0.87

    Issues breakdown:
      loop_detected:            3
      redundant_actions:        4
      low_diversity:            2
      impossible_coordinates:   0
    ══════════════════════════════════════════════
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

from visionnav.actions.recorder import ActionRecorder
from visionnav.actions.trajectory_analyzer import TrajectoryAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze recording quality for training data selection"
    )
    parser.add_argument("recordings_dir", type=Path)
    parser.add_argument(
        "--min-score", type=float, default=0.6,
        help="Minimum quality score to approve for training"
    )
    parser.add_argument(
        "--show-errors-only", action="store_true",
        help="Only show recordings with errors"
    )
    args = parser.parse_args()

    # Your implementation here:
    # 1. Find all *.jsonl files in recordings_dir
    # 2. For each file: load steps using ActionRecorder.load()
    # 3. Run TrajectoryAnalyzer.analyze()
    # 4. Print TrajectoryAnalyzer.format_report()
    # 5. Track summary statistics
    # 6. Print batch summary at the end


if __name__ == "__main__":
    main()
```

---

## Part D — Tests

Create `tests/unit/test_action_recorder.py`:

```python
import json
import numpy as np
import pytest
from pathlib import Path
from visionnav.actions.recorder import ActionRecorder, RecordedStep
from visionnav.actions.schema import Action, ActionType


def make_screenshot() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def make_action(action_type: ActionType = ActionType.CLICK) -> Action:
    return Action(
        type=action_type,
        coordinates=(0.5, 0.5) if action_type == ActionType.CLICK else None,
        description="test action",
    )


# Test 1: Recorder saves steps to JSONL
def test_recorder_saves_steps(tmp_path):
    recorder = ActionRecorder(
        task="Open Notepad",
        output_dir=tmp_path,
    )
    recorder.start()
    recorder.record_step(
        make_action(ActionType.KEY),
        make_screenshot(),
        "desktop text",
        {"width": 100, "height": 100},
    )
    recorder.record_step(
        make_action(ActionType.DONE),
        make_screenshot(),
        "notepad open",
        {"width": 100, "height": 100},
    )
    recorder.finish(success=True)

    assert recorder.output_path.exists()
    lines = recorder.output_path.read_text().strip().split("\n")
    assert len(lines) == 2   # two steps = two lines


# Test 2: Each line is valid JSON
def test_recorder_writes_valid_json(tmp_path):
    recorder = ActionRecorder(task="Test task", output_dir=tmp_path)
    recorder.start()
    recorder.record_step(
        make_action(), make_screenshot(), "text", {"width": 100, "height": 100}
    )
    recorder.finish()

    lines = recorder.output_path.read_text().strip().split("\n")
    for line in lines:
        data = json.loads(line)   # must not raise
        assert "step_index" in data
        assert "action" in data
        assert "task" in data


# Test 3: Screenshots are saved to disk
def test_recorder_saves_screenshots(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    recorder.start()
    recorder.record_step(
        make_action(), make_screenshot(), "", {"width": 100, "height": 100}
    )
    recorder.finish()

    screenshots = list(tmp_path.glob("screenshots/**/*.png"))
    assert len(screenshots) == 1


# Test 4: Context manager works correctly
def test_recorder_context_manager(tmp_path):
    with ActionRecorder(task="Test", output_dir=tmp_path) as recorder:
        recorder.record_step(
            make_action(), make_screenshot(), "", {"width": 100, "height": 100}
        )
    assert recorder.output_path.exists()


# Test 5: record_step without start raises
def test_record_without_start_raises(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        recorder.record_step(
            make_action(), make_screenshot(), "", {"width": 100, "height": 100}
        )


# Test 6: load() returns steps in order
def test_load_returns_steps_in_order(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    recorder.start()
    for i in range(5):
        recorder.record_step(
            make_action(), make_screenshot(), f"text {i}", {"width": 100, "height": 100}
        )
    recorder.finish()

    loaded = list(recorder.load(recorder.output_path))
    assert len(loaded) == 5
    assert [s.step_index for s in loaded] == [0, 1, 2, 3, 4]
```

Create `tests/unit/test_trajectory_analyzer.py`:

```python
import pytest
from visionnav.actions.recorder import RecordedStep
from visionnav.actions.trajectory_analyzer import (
    TrajectoryAnalyzer, IssueSeverity, QualityReport
)
from visionnav.actions.schema import ActionType
from datetime import datetime, timezone


def make_step(
    index: int,
    action_type: ActionType = ActionType.CLICK,
    coordinates: tuple = (0.5, 0.5),
) -> RecordedStep:
    action_dict = {"type": action_type.value}
    if action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
        action_dict["coordinates"] = list(coordinates)
    return RecordedStep(
        step_index    = index,
        task          = "test task",
        action        = action_dict,
        screenshot_path = f"step_{index:03d}.png",
        ocr_text      = "some text",
        screen_width  = 1920,
        screen_height = 1080,
        timestamp     = datetime.now(timezone.utc).isoformat(),
        session_id    = "test-session",
    )


# Test 1: Empty trajectory is flagged as error
def test_empty_trajectory_is_error():
    analyzer = TrajectoryAnalyzer()
    report   = analyzer.analyze([])
    assert report.quality_score == 0.0
    assert report.use_for_training is False
    assert any(i.severity == IssueSeverity.ERROR for i in report.issues)


# Test 2: Clean trajectory gets high score
def test_clean_trajectory_high_score():
    steps = [
        make_step(0, ActionType.KEY),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.CLICK),
        make_step(3, ActionType.DONE),
    ]
    report = TrajectoryAnalyzer().analyze(steps)
    assert report.quality_score >= 0.8
    assert report.use_for_training is True


# Test 3: Loop detection works
def test_loop_detection():
    # Pattern: click, type, click, type → loop of length 2
    steps = [
        make_step(0, ActionType.CLICK),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.CLICK),
        make_step(3, ActionType.TYPE),
        make_step(4, ActionType.DONE),
    ]
    analyzer = TrajectoryAnalyzer()
    issues   = analyzer.detect_action_loops(steps)
    assert len(issues) > 0
    assert any("loop" in i.issue_type.lower() for i in issues)


# Test 4: No false positive on non-looping trajectory
def test_no_loop_false_positive():
    steps = [
        make_step(0, ActionType.KEY),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.SCROLL),
        make_step(3, ActionType.CLICK),
        make_step(4, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_action_loops(steps)
    assert len(issues) == 0


# Test 5: Redundant actions detected
def test_redundant_actions_detected():
    # Two consecutive clicks at same coordinates
    steps = [
        make_step(0, ActionType.CLICK, (0.5, 0.5)),
        make_step(1, ActionType.CLICK, (0.5, 0.5)),
        make_step(2, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_redundant_actions(steps)
    assert len(issues) > 0


# Test 6: Low diversity flagged
def test_low_diversity_flagged():
    # 9 clicks and 1 done = 100% click ratio
    steps = [make_step(i, ActionType.CLICK) for i in range(9)]
    steps.append(make_step(9, ActionType.DONE))
    issues = TrajectoryAnalyzer().detect_low_diversity(steps)
    assert len(issues) > 0


# Test 7: Impossible coordinates flagged as error
def test_impossible_coordinates_flagged():
    steps = [
        make_step(0, ActionType.CLICK, (1.5, 0.5)),  # x > 1.0 ← impossible
        make_step(1, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_impossible_coordinates(steps)
    assert len(issues) > 0
    assert all(i.severity == IssueSeverity.ERROR for i in issues)


# Test 8: quality_score decreases with issues
def test_quality_score_decreases_with_issues():
    clean  = [make_step(i, t) for i, t in
              enumerate([ActionType.KEY, ActionType.TYPE, ActionType.CLICK, ActionType.DONE])]
    looped = [make_step(i, t) for i, t in
              enumerate([ActionType.CLICK, ActionType.TYPE,
                         ActionType.CLICK, ActionType.TYPE,
                         ActionType.DONE])]

    clean_score  = TrajectoryAnalyzer().analyze(clean).quality_score
    looped_score = TrajectoryAnalyzer().analyze(looped).quality_score
    assert clean_score > looped_score


# Test 9: format_report returns non-empty string
def test_format_report_returns_string():
    steps  = [make_step(0, ActionType.CLICK), make_step(1, ActionType.DONE)]
    report = TrajectoryAnalyzer().analyze(steps)
    text   = TrajectoryAnalyzer().format_report(report)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "Quality" in text
```

---


## How It All Works Together

```
Step 1 — Record:
  ActionRecorder captures agent actions as they run
  Each step: screenshot saved + action serialized → one JSONL line

Step 2 — Analyze:
  TrajectoryAnalyzer reads the JSONL file
  Runs 4 detectors → finds issues → computes quality score

Step 3 — Batch:
  analyze_recordings.py runs on ALL files in a directory
  Prints individual reports + aggregate summary
  Shows which recordings are safe for training

Step 4 — Training:
  Only approved recordings (score >= 0.6) enter the training pipeline
  Rejected recordings go to human review
```

Now run your tests and tell me the result.
---

## Success Criteria

```bash
python -m pytest tests/unit/test_action_recorder.py -v
python -m pytest tests/unit/test_trajectory_analyzer.py -v
python -m pytest tests/unit/ -v   # all existing tests still pass
```

---

## What I Will Review

```
ActionRecorder:
  □ Does load() use yield (generator) not return list?
  □ Does finish() use mode="a" (append not overwrite)?
  □ Does record_step raise clearly if start() not called?
  □ Are screenshots saved with step_index in filename?

TrajectoryAnalyzer:
  □ Is loop detection correct for windows of size 2 AND 3?
  □ Does low_diversity exclude DONE and FAIL from ratio?
  □ Does quality_score formula match the spec (0.3 per error)?
  □ Is format_report readable without being in the codebase?

Both:
  □ Type hints on every function
  □ Docstrings explain WHY not just WHAT
  □ No bare except: clauses

Connection to Phase 16:
  □ Do you understand how this feeds the Dataset Factory?
```

---

## One Thing To Understand Deeply

The `TrajectoryAnalyzer` you are building now is not just an exercise.

When we reach Phase 16 and start collecting thousands of real GUI recordings,
every single one will pass through this analyzer before becoming training data.

Bad recordings that slip through will teach the model wrong behaviors.
Good recordings that are falsely rejected waste expensive annotation effort.

The quality of this analyzer directly determines the quality of our dataset.
The quality of our dataset directly determines the quality of our model.
The quality of our model determines whether VisionNav succeeds or fails.

**This is not a coding exercise. This is a load-bearing component.**

Come back with all five assignments submitted when ready.