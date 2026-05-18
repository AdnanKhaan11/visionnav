"""Merge screenshot + OCR + UI tree into a single Observation."""
from __future__ import annotations
import base64
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from visionnav.perception.ocr import TextRegion


@dataclass
class Observation:
    screenshot_b64: str
    screenshot_path: str
    screen_width: int
    screen_height: int
    ocr_regions: list[TextRegion] = field(default_factory=list)
    ui_elements: list[dict]       = field(default_factory=list)
    platform: str                 = "desktop"

    def to_text_summary(self, max_ocr: int = 30) -> str:
        lines: list[str] = []
        if self.ocr_regions:
            lines.append("Detected text on screen:")
            for r in self.ocr_regions[:max_ocr]:
                x1, y1, x2, y2 = r.bbox
                lines.append(f"  - '{r.text}' at [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]")
        if self.ui_elements:
            lines.append("\nInteractive elements:")
            for el in self.ui_elements[:20]:
                lines.append(
                    f"  - [{el.get('type','?')}] '{el.get('label','')}'"
                    f" at {el.get('bounds','?')}"
                )
        return "\n".join(lines) if lines else "Screen appears empty or unreadable."


def fuse(
    image: np.ndarray,
    meta: dict,
    ocr_regions: list[TextRegion],
    ui_elements: list[dict],
) -> Observation:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return Observation(
        screenshot_b64=b64,
        screenshot_path=meta.get("path", ""),
        screen_width=meta.get("width", image.shape[1]),
        screen_height=meta.get("height", image.shape[0]),
        ocr_regions=ocr_regions,
        ui_elements=ui_elements,
        platform=meta.get("platform", "desktop"),
    )
