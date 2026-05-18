"""Desktop automation — Windows / macOS / Linux via pyautogui + mss."""
from __future__ import annotations
import asyncio
import numpy as np
import pyautogui
import structlog
from visionnav.perception.capture import ScreenCapture
from visionnav.perception.ui_tree import get_ui_tree
from visionnav.platforms.base import PlatformAdapter

log = structlog.get_logger(__name__)
pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.05


class DesktopPlatform(PlatformAdapter):
    def __init__(self, monitor_index: int = 1) -> None:
        self._capture = ScreenCapture(monitor_index)

    async def capture(self) -> tuple[np.ndarray, dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._capture.capture)

    async def get_ui_tree(self) -> list[dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, get_ui_tree)

    def get_screen_size(self) -> tuple[int, int]:
        return self._capture.get_screen_size()

    async def execute_click(self, x: int, y: int, button: str = "left") -> bool:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.click(x, y, button=button))
        return True

    async def execute_type(self, text: str) -> bool:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.write(text, interval=0.03))
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        clicks = amount if direction == "up" else -amount
        loop   = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.scroll(clicks, x=x, y=y))
        return True

    async def execute_key(self, key_combo: str) -> bool:
        keys = key_combo.lower().split("+")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))
        return True
