"""Dispatch typed Actions to the platform adapter.

This module is the AI agent’s:

“Action Execution Engine”
Its job is:
take a parsed Action object and actually perform it on the computer.
"""

from __future__ import annotations
import asyncio
import structlog
from visionnav.actions.schema import Action, ActionType
from visionnav.platforms.base import PlatformAdapter
from visionnav.utils.coords import denormalize

log = structlog.get_logger(__name__)


class ActionExecutor:
    def __init__(self, platform: PlatformAdapter) -> None:
        self._platform = platform

    async def execute(self, action: Action, screen_w: int, screen_h: int) -> bool:
        try:
            return await self._dispatch(action, screen_w, screen_h)
        except Exception as exc:
            log.error("execute_failed", action=action.type, error=str(exc))
            return False

    # decide which action to execute. For example, if it's a click, we call the platform's click method with the appropriate coordinates. If it's a type action, we call the platform's type method with the text to type. And so on for each action type.
    async def _dispatch(self, action: Action, w: int, h: int) -> bool:
        # Modern Python switch-case.
        match action.type:
            case ActionType.CLICK:
                x, y = self._abs(action, w, h)
                # Actually clicks mouse. Note that we use "left" for click and double-click, and "right" for right-click. This is a common convention, but it could be made more flexible if needed.
                return await self._platform.execute_click(x, y, "left")
            case ActionType.DOUBLE_CLICK:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_click(x, y, "left")
            case ActionType.RIGHT_CLICK:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_click(x, y, "right")
            case ActionType.TYPE:
                return await self._platform.execute_type(action.text or "")
            case ActionType.KEY:
                return await self._platform.execute_key(action.key or "")
            case ActionType.SCROLL:
                x, y = self._abs(action, w, h)
                return await self._platform.execute_scroll(
                    x, y, action.direction or "down", action.amount or 3
                )
            case ActionType.WAIT:
                await asyncio.sleep(action.duration_ms / 1000.0)
                return True
            case ActionType.DONE | ActionType.FAIL | ActionType.SCREENSHOT:
                return True
            case _:
                log.warning("unknown_action", type=action.type)
                return False

    def _abs(self, action: Action, w: int, h: int) -> tuple[int, int]:
        if not action.coordinates:
            raise ValueError(f"{action.type} requires coordinates")
        return denormalize(action.coordinates[0], action.coordinates[1], w, h)


# ============================================================
# ActionExecutor
# ============================================================
# This module is the "hands" of the AI agent.
#
# Flow:
# AI Model Output
#      ↓
# Parser creates Action object
#      ↓
# ActionExecutor executes real computer actions
#
# Main responsibilities:
# - Decide which action to execute (CLICK, TYPE, SCROLL, etc.)
# - Convert normalized coordinates (0→1) into real screen pixels
# - Call platform functions that control mouse/keyboard
# - Handle execution errors safely
#
# Why normalized coordinates?
# The AI outputs positions as percentages so actions work
# on different screen sizes. Example:
#   (0.5, 0.5) = center of any screen
#
# Before clicking, coordinates are converted back into
# real pixels because the operating system only understands
# actual pixel positions.
#
# match-case acts like a dispatcher:
# CLICK  → click function
# TYPE   → keyboard typing function
# SCROLL → scrolling function
#
# asyncio.sleep() is used for WAIT actions so the agent
# can pause while apps/pages load.
# ============================================================
