"""Merge screenshot + OCR + UI tree into a single Observation."""

from __future__ import annotations

import asyncio
import base64
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


from visionnav.perception.ocr import TextRegion
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visionnav.perception.ocr import OCREngine
    from visionnav.platforms.base import PlatformAdapter


# Main container holding everything the agent observes.
@dataclass
class Observation:
    screenshot_b64: str
    screenshot_path: str
    screen_width: int
    screen_height: int
    ocr_regions: list[TextRegion] = field(default_factory=list)
    ui_elements: list[dict] = field(default_factory=list)
    platform: str = "desktop"

    # Creates human-readable text summary because LLMs understand
    # text better than raw structures.
    def to_text_summary(self, max_ocr: int = 30) -> str:
        lines: list[str] = []

        if self.ocr_regions:
            lines.append("Detected text on screen:")

            for r in self.ocr_regions[:max_ocr]:
                x1, y1, x2, y2 = r.bbox

                lines.append(
                    f"  - '{r.text}' at " f"[{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]"
                )

        if self.ui_elements:
            lines.append("\nInteractive elements:")

            # Limit UI elements to keep summaries compact.
            for el in self.ui_elements[:20]:
                lines.append(
                    f"  - [{el.get('type', '?')}] "
                    f"'{el.get('label', '')}' "
                    f"at {el.get('bounds', '?')}"
                )

        return "\n".join(lines) if lines else "Screen appears empty or unreadable."


# Merge image, OCR, and UI data into a unified Observation.
def fuse(
    image: np.ndarray,
    meta: dict,
    ocr_regions: list[TextRegion],
    ui_elements: list[dict],
) -> Observation:
    """
    Combine screenshot, OCR regions, and UI tree into a single Observation.

    Why base64?
    The screenshot may later be sent to APIs or stored in JSON,
    so we convert the PNG bytes into a portable text format.
    """

    # Convert NumPy array → PIL image.
    pil = Image.fromarray(image)

    # Store image in memory instead of writing temporary file.
    buf = io.BytesIO()

    # Save as compressed PNG.
    pil.save(buf, format="PNG", optimize=True)

    # Convert image bytes → base64 string.
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


async def fuse_parallel(
    platform: PlatformAdapter,
    ocr_engine: OCREngine,
    meta_override: dict | None = None,
) -> Observation:
    """
    Capture screenshot and UI tree concurrently using asyncio.gather,
    then run OCR on the captured screenshot.

    Why parallel?
    Screenshot capture and UI tree extraction are independent I/O tasks.
    Running them together reduces total perception latency.

    Sequential:
        capture (50ms) + ui tree (30ms) = 80ms

    Parallel:
        max(50ms, 30ms) = ~50ms

    This improves responsiveness for every agent step.
    """

    # Run both async operations concurrently.
    (image, meta), ui_elements = await asyncio.gather(
        platform.capture(),
        platform.get_ui_tree(),
    )

    # OCR is synchronous CPU work.
    # It must happen AFTER screenshot capture because OCR needs the image.
    ocr_regions = ocr_engine.run(image)

    # Merge optional metadata overrides.
    if meta_override:
        meta = {**meta, **meta_override}

    # Combine everything into Observation.
    observation = fuse(
        image=image,
        meta=meta,
        ocr_regions=ocr_regions,
        ui_elements=ui_elements,
    )

    return observation


# def to_text_summary(self, max_ocr: int = 30) -> str:
# Creates human-readable text summary.

# Why Needed?
# LLMs understand text better than raw structures.
# So convert observation into readable summary.

# field(default_factory=list)
# ────────────────────────────────────────────
#
# Creates a NEW empty list for every object instance.
#
# Example:
#
# @dataclass
# class Test:
#     items: list = field(default_factory=list)
#
# a = Test()
# b = Test()
#
# a.items.append("hello")
#
# print(a.items)   # ['hello']
# print(b.items)   # []
#
# Each object gets its OWN separate list.
#
#
# WHY MUTABLE DEFAULTS ARE DANGEROUS
# ────────────────────────────────────────────
#
# This is dangerous:
#
# @dataclass
# class Test:
#     items: list = []
#
# Because the SAME list is shared between ALL objects.
#
# Example:
#
# a = Test()
# b = Test()
#
# a.items.append("hello")
#
# print(b.items)
# # ['hello']  ← unexpected shared data!
#
#
# Mutable means:
# object can change after creation.
#
# Lists, dicts, and sets are mutable objects.
# ────────────────────────────────────────────


# WHAT IS BytesIO ?
# ────────────────────────────────────────────
#
# BytesIO is an in-memory binary file.
#
# Think of it like:
#
# "fake file stored in RAM instead of disk"
#
#
# Normal file:
#
# with open("img.png", "wb") as f:
#     f.write(data)
#
# Saves data to disk.
#
#
# BytesIO:
#
# buf = io.BytesIO()
# buf.write(data)
#
# Saves data in memory (RAM), not on disk.
#
#
# WHY USE IT?
# ────────────────────────────────────────────
#
# Faster than creating temporary files.
#
# Useful when:
# - processing images
# - sending API data
# - encoding files
#
#
# WHAT IS "VIRTUAL FILE"?
# ────────────────────────────────────────────
#
# It behaves like a real file:
#
# - write()
# - read()
# - save()
#
# but exists only in RAM memory.
#
# No physical file is created.
# ────────────────────────────────────────────


# base64.b64encode(buf.getvalue()).decode()
# ────────────────────────────────────────────
#
# Converts binary image bytes into readable text format.
#
#
# Step-by-step:
#
# 1. buf.getvalue()
#
# Get raw binary bytes from memory buffer.
#
#
# Example:
#
# b'\x89PNG\x00...'
#
#
# 2. base64.b64encode(...)
#
# Convert binary bytes → Base64 encoded bytes.
#
#
# Example:
#
# b'iVBORw0KGgoAAAANS...'
#
#
# 3. .decode()
#
# Convert bytes → normal Python string.
#
#
# Final result:
#
# "iVBORw0KGgoAAAANS..."
#
#
# WHY USE BASE64?
# ────────────────────────────────────────────
#
# Binary image data cannot be safely stored in:
#
# - JSON
# - APIs
# - text databases
#
#
# Base64 converts binary data into safe text format
# that can be transmitted/stored easily.
# ────────────────────────────────────────────
