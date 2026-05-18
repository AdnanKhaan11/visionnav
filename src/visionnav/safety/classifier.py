"""Risk classifier — assign a RiskLevel to each action before execution."""
from __future__ import annotations
from enum import IntEnum
from visionnav.actions.schema import Action, ActionType


class RiskLevel(IntEnum):
    SAFE    = 0
    LOW     = 1
    MEDIUM  = 2
    HIGH    = 3
    BLOCKED = 4


_BASE: dict[ActionType, RiskLevel] = {
    ActionType.SCREENSHOT: RiskLevel.SAFE,
    ActionType.WAIT:       RiskLevel.SAFE,
    ActionType.SCROLL:     RiskLevel.SAFE,
    ActionType.DONE:       RiskLevel.SAFE,
    ActionType.FAIL:       RiskLevel.SAFE,
    ActionType.CLICK:      RiskLevel.LOW,
    ActionType.DOUBLE_CLICK: RiskLevel.LOW,
    ActionType.RIGHT_CLICK:  RiskLevel.LOW,
    ActionType.KEY:        RiskLevel.LOW,
    ActionType.DRAG:       RiskLevel.LOW,
    ActionType.SWIPE:      RiskLevel.LOW,
    ActionType.LONG_PRESS: RiskLevel.LOW,
    ActionType.TYPE:       RiskLevel.MEDIUM,
}
_HIGH_CLICK = {"delete","remove","buy","purchase","send","pay","confirm","checkout","uninstall"}
_HIGH_TYPE  = {"password","card","cvv","ssn","secret","token"}


class SafetyClassifier:
    def classify(self, action: Action, context: str = "") -> RiskLevel:
        base = _BASE.get(action.type, RiskLevel.MEDIUM)
        ctx  = (action.description + " " + context).lower()
        if action.type == ActionType.CLICK and any(k in ctx for k in _HIGH_CLICK):
            return RiskLevel.HIGH
        if action.type == ActionType.TYPE and any(k in ctx for k in _HIGH_TYPE):
            return RiskLevel.HIGH
        return base
