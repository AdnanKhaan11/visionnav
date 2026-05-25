"""AgentState — immutable snapshot of one agent step."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from visionnav.actions.schema import Action


@dataclass(
    frozen=True
)  # What does frozen=True mean? Makes object IMMUTABLE. Once created, its fields cannot be changed. This is useful for ensuring that the state remains consistent and prevents accidental modifications.
class AgentState:
    step_index: int
    task_instruction: str  # Original instruction for the task, e.g. "Open the calculator app and calculate 2+2"
    screenshot_path: str
    ocr_text: str
    action_taken: Optional["Action"]  # What action was executed.
    action_success: bool
    reasoning: str  # AI thinking text
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = None

    def to_history_entry(self) -> dict:
        action_str = ""
        if self.action_taken:
            action_str = f"{self.action_taken.type} | success={self.action_success}"
        return {"role": "assistant", "content": f"Step {self.step_index}: {action_str}"}


@dataclass
class TaskResult:
    task_id: str
    success: bool
    steps: int
    error: Optional[str] = None
    summary: str = ""
    elapsed_s: float = 0.0


# 1. What is “Memory Snapshot”?
# A memory snapshot is:
# a saved picture of what happened at one moment.
# example:
# Step 3:
# - Saw Chrome icon
# - Clicked it
# - Success = True
#  The agent saves this information so it can remember previous actions.


# 2. What does “Represents ONE step memory” mean?
# It means:
# one object stores information about ONLY ONE agent step.

# Parsing means:

# converting raw text/data into a structured format that the program can understand.
