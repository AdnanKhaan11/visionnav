"""Task Reporter — human-readable summary from step history."""
from __future__ import annotations
from visionnav.agent.state import AgentState, TaskResult


class TaskReporter:
    def generate(self, result: TaskResult, history: list[AgentState]) -> str:
        status = "SUCCESS" if result.success else "FAILED"
        lines = [
            f"Task Report — {status}",
            "=" * 50,
            f"Task ID : {result.task_id}",
            f"Steps   : {result.steps}",
            f"Outcome : {result.summary or result.error or status}",
            "=" * 50,
            "Step Log:",
        ]
        for s in history:
            info = "—"
            if s.action_taken:
                info = f"{s.action_taken.type} {'OK' if s.action_success else 'FAIL'}"
            lines.append(f"  [{s.step_index:02d}] {info}")
            if s.error:
                lines.append(f"       Error: {s.error}")
        return "
".join(lines)
