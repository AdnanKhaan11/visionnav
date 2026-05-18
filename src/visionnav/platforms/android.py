"""Android automation via ADB."""
from __future__ import annotations
import io
import numpy as np
import structlog
from visionnav.platforms.base import PlatformAdapter

log = structlog.get_logger(__name__)


class AndroidPlatform(PlatformAdapter):
    def __init__(self, serial: str | None = None) -> None:
        import adbutils
        self._device = adbutils.adb.device(serial=serial)
        log.info("android_connected", serial=self._device.serial)

    async def capture(self) -> tuple[np.ndarray, dict]:
        from PIL import Image
        img  = Image.open(io.BytesIO(self._device.screencap())).convert("RGB")
        arr  = np.array(img)
        meta = {"width": img.width, "height": img.height, "platform": "android"}
        return arr, meta

    async def get_ui_tree(self) -> list[dict]:
        return []   # TODO Phase 5: UIAutomator2 dump

    def get_screen_size(self) -> tuple[int, int]:
        info = self._device.window_size()
        return info.width, info.height

    async def execute_click(self, x: int, y: int, button: str = "left") -> bool:
        self._device.shell(f"input tap {x} {y}")
        return True

    async def execute_type(self, text: str) -> bool:
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        self._device.shell(f"input text '{escaped}'")
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        dist = amount * 300
        if direction == "down":
            self._device.shell(f"input swipe {x} {y} {x} {y-dist} 300")
        else:
            self._device.shell(f"input swipe {x} {y} {x} {y+dist} 300")
        return True

    async def execute_key(self, key_combo: str) -> bool:
        kmap = {"enter":"66","back":"4","home":"3","tab":"61"}
        kc   = kmap.get(key_combo.lower(), "")
        if kc:
            self._device.shell(f"input keyevent {kc}")
        return bool(kc)
