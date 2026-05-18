"""Task Planner — decomposes a task string into ordered step descriptions.
Summary of this module is :
“The AI Agent’s Basic Planner”
It:
reads user task
finds matching task type
returns ordered mini-steps
helps reasoning system stay organized
"""

from __future__ import annotations
import structlog

log = structlog.get_logger(__name__)

_TASK_TEMPLATES: dict[str, list[str]] = {
    "open": [
        "Find the target app or file",
        "Click to open it",
        "Wait for it to load",
        "Verify it opened successfully",
    ],
    "search": [
        "Open the search interface",
        "Click the search field",
        "Type the search query",
        "Press Enter",
        "Wait for results",
    ],
    "login": [
        "Navigate to the login page",
        "Enter username",
        "Enter password",
        "Click login",
        "Verify login succeeded",
    ],
    "fill": [
        "Identify all form fields",
        "Fill each field in order",
        "Review the filled values",
        "Submit the form",
        "Verify submission",
    ],
    "click": ["Locate the target element", "Click on it", "Verify the action worked"],
    "type": [
        "Find the input field",
        "Click to focus it",
        "Type the text",
        "Verify text was entered",
    ],
    "scroll": [
        "Determine scroll direction",
        "Scroll to reveal content",
        "Verify target is visible",
    ],
}

_GENERIC_PLAN = [
    "Analyse the current screen state",
    "Identify the action needed for the task",
    "Execute the action",
    "Verify the result",
    "Continue until task is complete",
]


class TaskPlanner:
    def decompose(self, task: str) -> list[str]:
        task_lower = task.lower()
        for keyword, steps in _TASK_TEMPLATES.items():
            if keyword in task_lower:
                log.debug("plan_matched", keyword=keyword, steps=len(steps))
                return steps
        log.debug("plan_generic", task=task[:60])
        return _GENERIC_PLAN
