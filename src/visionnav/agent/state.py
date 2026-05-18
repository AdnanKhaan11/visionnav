"""AgentState — immutable snapshot of one agent step."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from visionnav.actions.schema import Action


@dataclass(frozen=True)
class AgentState:
    step_index: int
    task_instruction: str
    screenshot_path: str
    ocr_text: str
    action_taken: Optional["Action"]
    action_success: bool
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
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
