"""Cross-platform screenshot capture using mss."""
from __future__ import annotations
import numpy as np
import mss


class ScreenCapture:
    def __init__(self, monitor_index: int = 1) -> None:
        self._monitor_index = monitor_index

    def capture(self) -> tuple[np.ndarray, dict]:
        with mss.mss() as sct:
            mon  = sct.monitors[self._monitor_index]
            shot = sct.grab(mon)
            arr  = np.frombuffer(shot.rgb, dtype=np.uint8)
            arr  = arr.reshape(shot.height, shot.width, 3)
            meta = {"width": shot.width, "height": shot.height,
                    "monitor": self._monitor_index}
        return arr, meta

    def get_screen_size(self) -> tuple[int, int]:
        with mss.mss() as sct:
            m = sct.monitors[self._monitor_index]
            return m["width"], m["height"]
