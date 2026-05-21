"""Cross-platform screenshot capture using mss."""

from __future__ import annotations

import numpy as np
import mss


class ScreenCapture:
    """
    Captures the screen as a numpy RGB array.
    Uses mss — pure Python, fast, works on Windows/macOS/Linux.

    """

    def __init__(self, monitor_index: int = 1) -> None:
        self._monitor_index = monitor_index

    def capture(self) -> tuple[np.ndarray, dict]:
        """
        Capture screenshot and return as numpy array.

        Returns:
            arr  : numpy array (H, W, 3) uint8 RGB
            meta : dict with width, height, monitor index
        """
        with mss.MSS() as sct:
            mon = sct.monitors[
                self._monitor_index
            ]  # Gets monitor configuration. monitor info contain width, height, top.left coordinates of the monitor. We use this to capture the correct screen area.

            # Takes screenshot. mss returns raw bytes, so we convert to numpy array and reshape to (height, width, 3).
            shot = sct.grab(mon)

            # Converts raw bytes → NumPy array.
            arr = np.frombuffer(shot.rgb, dtype=np.uint8)
            arr = arr.reshape(shot.height, shot.width, 3)
            meta = {
                "width": shot.width,
                "height": shot.height,
                "monitor": self._monitor_index,
            }
        return arr, meta

    def capture_normalized(
        self,
        target_w: int = 1280,
        target_h: int = 720,
    ) -> tuple[np.ndarray, dict]:
        """
        Capture and resize to standard resolution.

        Why normalize?
        All screenshots must be same size for consistent model input.
        A 1920x1080 screen and a 1280x720 screen both become 1280x720.
        This means coordinates are always in the same scale during training.
        """
        #  meta deta means Extra info about screenshot. It includes width, height, monitor index, and whether it was resized.
        arr, meta = self.capture()

        if arr.shape[1] != target_w or arr.shape[0] != target_h:
            from visionnav.utils.image import resize_to_target

            arr = resize_to_target(arr, target_w, target_h)
            meta["width"] = target_w
            meta["height"] = target_h
            meta["resized"] = True

        return arr, meta

    def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the monitored screen in pixels."""
        with mss.MSS() as sct:
            m = sct.monitors[self._monitor_index]
            return m["width"], m["height"]


# capture.py
#  monitor_index:
#      Which monitor to capture.
#      Example: If user has 2 monitors and wants to capture the second one, set monitor_index=2.
#      If user has:
#      monitor 1
#      monitor 2
#      you can choose.

# WHY "with" ?
# ────────────────────────────────────────────
#
# "with" automatically cleans resources after use.
#
# Think of it like:
#
# "Open → Use → Close automatically"
#
#
# Example without "with":
#
# file = open("data.txt")
# data = file.read()
# file.close()   # must close manually
#
#
# Example with "with":
#
# with open("data.txt") as file:
#     data = file.read()
#
# File automatically closes after block ends.
#
#
# Same idea here:
#
# with mss.MSS() as sct:
#
# Python automatically cleans/closes the screenshot session.
#
# This prevents:
# - memory leaks
# - resource issues
# - locked handles
#
# ────────────────────────────────────────────


# WHAT IS shot.rgb ?
# ────────────────────────────────────────────
#
# shot.rgb contains raw image pixel bytes.
#
# Computers store images as numbers:
#
# RGB = Red Green Blue
#
# Example:
#
# Red pixel:
# (255, 0, 0)
#
# Green pixel:
# (0, 255, 0)
#
# Blue pixel:
# (0, 0, 255)
#
#
# shot.rgb is basically a long stream of pixel values:
#
# [255,0,0, 0,255,0, 0,0,255 ...]
#
# These are raw image bytes from the screenshot.
#
# We later convert them into a NumPy image array.
#
# ────────────────────────────────────────────


# with mss.MSS() as sct:
# ────────────────────────────────────────────
#
# Creates a screenshot session using MSS library.
#
# Think of it like:
#
# "Start screenshot tool"
#
# Example flow:
#
# with mss.MSS() as sct:
#     shot = sct.grab(...)
#
#
# Step-by-step:
#
# 1. Start MSS screenshot system
# 2. Capture screenshot
# 3. Automatically clean resources
#
#
# "sct" is just a variable name.
#
# Similar to:
#
# with open("a.txt") as file:
#
# Here:
#
# file = opened file object
#
# In MSS:
#
# sct = screenshot session object
#
# ────────────────────────────────────────────
