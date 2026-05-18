"""Compare before/after screenshots to detect whether an action had effect."""
from __future__ import annotations
import numpy as np
from visionnav.actions.schema import Action, ActionType

_NO_VERIFY = {ActionType.DONE, ActionType.FAIL, ActionType.WAIT, ActionType.SCREENSHOT}


class ActionVerifier:
    def __init__(self, change_threshold: float = 0.01) -> None:
        self._threshold = change_threshold

    def verify(
        self, before: np.ndarray, after: np.ndarray, action: Action
    ) -> tuple[bool, float]:
        if action.type in _NO_VERIFY:
            return True, 0.0
        if before.shape != after.shape:
            return True, 1.0
        diff         = np.abs(before.astype(np.int16) - after.astype(np.int16))
        change_ratio = float((diff > 30).mean())
        return change_ratio >= self._threshold, change_ratio
