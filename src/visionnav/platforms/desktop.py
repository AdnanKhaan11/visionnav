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
pyautogui.FAILSAFE = True  # If mouse moves to top-left corner: stop the program. why we do this is because if the AI goes rogue and starts clicking randomly, you can quickly move your mouse to the top-left corner to stop it. This is a safety feature to prevent unintended consequences.
pyautogui.PAUSE = 0.05


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
        #  loop.run_in_executor() here is loop is the current event loop, (event loop simply means its the thing that runs async functions and manages their execution how it manage execution ? it allows multiple async functions to run concurrently without blocking each other. when you call loop.run_in_executor()  run_in_executor() is a method that allows you to run a synchronous function (like pyautogui.click) in a separate thread or process, so it doesn't block the main event loop. the first argument None means to use the default executor (which is usually a thread pool), and the second argument is a lambda function that calls pyautogui.click with the specified x, y, and button parameters. this way, the click action will be performed without blocking other async tasks from running concurrently.
        # Event loop coordinates async tasks.It decides: which task run now ? which task run next ? when a task is waiting for something (like a click to finish), the event loop can switch to another task that is ready to run. this allows for efficient multitasking and keeps the application responsive.
        await loop.run_in_executor(None, lambda: pyautogui.click(x, y, button=button))
        return True

    async def execute_type(self, text: str) -> bool:
        # get_event_loop() is a method that returns the current event loop. An event loop is a programming construct(construct means a mechanism or tool) that waits for and dispatches events or messages in a program. In the context of asyncio, the event loop is responsible for managing and executing asynchronous tasks. When you call get_event_loop(), it gives you access to the event loop that is currently running, allowing you to schedule tasks or run functions in an asynchronous manner.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.write(text, interval=0.03))
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        # amount is the number of scroll steps, scroll step means how much to scroll with each step. direction is either "up" or "down". if direction is "up", we want to scroll up by the specified amount, which means we need to pass a positive value to pyautogui.scroll. if direction is "down", we want to scroll down by the specified amount, which means we need to pass a negative value to pyautogui.scroll. so we can calculate the number of
        clicks = amount if direction == "up" else -amount
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: pyautogui.scroll(clicks, x=x, y=y))
        return True

    async def execute_key(self, key_combo: str) -> bool:
        keys = key_combo.lower().split("+")
        loop = asyncio.get_event_loop()
        #  the lambda function is return a function that calls pyautogui.hotkey (.hotkey means to press multiple keys at the same time,for example if key_combo is "ctrl+c", key will be ["ctrl","c" and  pyautogui.hotkey("ctrl","c") will simulate pressing the "ctrl" key and the "c" key at the same time, which is the keyboard shortcut for copying. by using run_in_executor, we can run this synchronous function in a separate thread, allowing it to execute without blocking the main event loop.)
        await loop.run_in_executor(None, lambda: pyautogui.hotkey(*keys))
        return True


# This module is the “hands and eyes”
# of your AI agent on desktop computers.

# It allows the AI to:

# - see the screen
# - inspect UI
# - click things
# - type text
# - scroll
# - press keyboard shortcuts

# while hiding OS-specific complexity.
