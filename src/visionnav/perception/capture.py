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
            mon = sct.monitors[self._monitor_index]
            shot = sct.grab(mon)
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
