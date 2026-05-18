"""Parse raw model text → typed Action. Defensive against all malformed output."""
from __future__ import annotations
import json
import re
from visionnav.actions.schema import Action, ActionType

_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)


class ActionParseError(ValueError):
    """Raised when model output cannot be parsed into a valid Action."""


def parse_action(model_output: str) -> Action:
    match = _PATTERN.search(model_output)
    if not match:
        raise ActionParseError(f"No <action> block found. Preview: {model_output[:200]!r}")

    try:
        raw = json.loads(match.group(1).strip())
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
