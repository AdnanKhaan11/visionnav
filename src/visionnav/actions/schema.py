"""Typed Action schema."""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    CLICK        = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK  = "right_click"
    LONG_PRESS   = "long_press"
    TYPE         = "type"
    KEY          = "key"
    SCROLL       = "scroll"
    SWIPE        = "swipe"
    DRAG         = "drag"
    WAIT         = "wait"
    SCREENSHOT   = "screenshot"
    DONE         = "done"
    FAIL         = "fail"


class Action(BaseModel):
    type:        ActionType
    coordinates: Optional[tuple[float, float]] = None
    text:        Optional[str]  = None
    key:         Optional[str]  = None
    direction:   Optional[str]  = None
    amount:      int            = 0
    duration_ms: int            = 0
    description: str            = ""
    confidence:  float          = Field(default=1.0, ge=0.0, le=1.0)
