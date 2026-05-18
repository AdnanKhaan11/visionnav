"""Dispatch typed Actions to the platform adapter."""
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

    async def _dispatch(self, action: Action, w: int, h: int) -> bool:
        match action.type:
            case ActionType.CLICK:
                x, y = self._abs(action, w, h)
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
                    x, y, action.direction or "down", action.amount or 3)
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
