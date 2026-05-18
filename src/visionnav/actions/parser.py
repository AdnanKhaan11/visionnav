"""Parse raw model text → typed Action. Defensive against all malformed output.
Means:
convert raw AI text into safe Action object.
“Defensive” means:
protect against broken AI output.
Very important because LLMs often generate garbage.
"""

from __future__ import annotations
import json
import re
from visionnav.actions.schema import Action, ActionType

_PATTERN = re.compile(
    r"<action>(.*?)</action>", re.DOTALL
)  # re.DOTALL allows the .*? to match newlines, so it can capture multi-line JSON content between the <action> tags.


class ActionParseError(ValueError):
    """Raised when model output cannot be parsed into a valid Action."""


def parse_action(model_output: str) -> Action:
    match = _PATTERN.search(model_output)
    if not match:
        raise ActionParseError(
            f"No <action> block found. Preview: {model_output[:200]!r}"
        )

    try:
        raw = json.loads(
            match.group(1).strip()
        )  # match.group(1)? Yes, because the regex has one capturing group (the parentheses around .*?), so match.group(1) gives us the content inside the <action>...</action> tags. We also call .strip() to remove any leading/trailing whitespace before parsing it as JSON.

    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON: {exc}") from exc

    try:
        action_type = ActionType(raw.get("type", ""))
    except ValueError:
        raise ActionParseError(f"Unknown action type: {raw.get('type')!r}")

    coords = raw.get("coordinates")
    if coords is not None:
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ActionParseError(f"coordinates must be [x, y], got: {coords!r}")
        # Ensures numbers are numeric. Also converts ints to floats, which is important for downstream processing that expects floats.
        nx, ny = float(coords[0]), float(coords[1])
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise ActionParseError(f"coordinates out of [0,1]: [{nx},{ny}]")
        coords = (nx, ny)

    return Action(
        type=action_type,
        coordinates=coords,
        text=raw.get("text"),
        key=raw.get("key"),
        direction=raw.get("direction"),
        amount=int(raw.get("amount", 0)),
        duration_ms=int(raw.get("duration_ms", 0)),
        description=str(raw.get("description", "")),
        confidence=float(raw.get("confidence", 1.0)),
    )
