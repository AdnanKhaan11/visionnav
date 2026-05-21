"""OCR Engine — PaddleOCR primary, Tesseract fallback."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class TextRegion:
    text: str
    bbox: tuple[float, float, float, float]  # normalised (x1,y1,x2,y2)
    confidence: float


class OCREngine:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_confidence = min_confidence
        self._paddle = None  # lazy-loaded means Do NOT load heavy model immediately.load when needed

    def run(self, image: np.ndarray) -> list[TextRegion]:
        h, w = image.shape[:2]
        try:
            return self._run_tesseract(image, w, h)
        except Exception as exc:
            log.warning("ocr_failed", error=str(exc))
            return []

    def _run_paddle(self, image: np.ndarray, w: int, h: int) -> list[TextRegion]:
        if self._paddle is None:
            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        results = self._paddle.ocr(image, cls=True)
        regions: list[TextRegion] = []
        if not results or not results[0]:  # If OCR found nothing.
            return regions
        for line in results[0]:
            pts, (text, conf) = line
            if conf < self._min_confidence:
                continue
            xs = [
                p[0] for p in pts
            ]  # why we use pts[0] and pts[1] instead of directly using pts? Because pts is a list of points that define the bounding box of the detected text. Each point is a tuple (x, y). To calculate the bounding box, we need to find the minimum and maximum x and y coordinates from these points. Therefore, we extract the x and y coordinates into separate lists (xs and ys) to easily compute the bounding box.
            ys = [p[1] for p in pts]
            regions.append(
                TextRegion(
                    text=text.strip(),
                    bbox=(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h),
                    confidence=float(conf),
                )
            )
        return regions

    def _run_tesseract(self, image: np.ndarray, w: int, h: int) -> list[TextRegion]:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = (
            r"D:\mlops-tools\Tesseract-OCR\tesseract.exe"
        )
        from PIL import Image

        data = pytesseract.image_to_data(
            Image.fromarray(image), output_type=pytesseract.Output.DICT
        )
        regions: list[TextRegion] = []
        for i, text in enumerate(data["text"]):
            conf = int(data["conf"][i])
            if conf < int(self._min_confidence * 100) or not text.strip():
                continue
            x, y, bw, bh = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            regions.append(
                TextRegion(
                    text=text.strip(),
                    bbox=(x / w, y / h, (x + bw) / w, (y + bh) / h),
                    confidence=conf / 100.0,
                )
            )
        # Filter out noise — single characters and symbols
        regions = [
            r
            for r in regions
            if len(r.text.strip()) > 2
            and r.confidence > 0.6
            and not r.text.strip().isnumeric()
            and "{" not in r.text
            and "$" not in r.text
            and len([c for c in r.text if c.isalpha()]) > 1
        ]
        return regions


# what is x w,y,b,h and why we use it here ?
# x and y are the top-left coordinates of the bounding box,
#  bw and bh are the width and height of the bounding box.
# We use these values to calculate the bottom-right coordinates
# of the bounding box by adding bw to x and bh to y. Then we
#  normalise these coordinates by dividing by the width (w)
# and height (h) of the image to get values between 0 and 1, which
# represent the relative position of the bounding box within the image.

# example of this .
# ─────────────────────────────────────────────────────────────────────
# BOUNDING BOX EXPLANATION
# ─────────────────────────────────────────────────────────────────────
#
# OCR detects text by drawing a rectangle around the text area.
# This rectangle is called a "Bounding Box".
#
# Example:
#
#                 IMAGE
#
#   (0,0)
#      ↓
#      ┌──────────────────────────────────────┐
#      │                                      │
#      │                                      │
#      │        ┌────────────────┐            │
#      │        │     LOGIN      │            │
#      │        └────────────────┘            │
#      │         ↑              ↑             │
#      │         │<---- bw ---->│             │
#      │                                      │
#      └──────────────────────────────────────┘
#                              →
#                             x-axis
#
#
# Bounding Box Variables
# ────────────────────────────────────────────
#
# x  = left position of box
# y  = top position of box
#
# bw = bounding box width
# bh = bounding box height
#
#
# Top-left corner:
# ────────────────────────────────────────────
#
# (x, y)
#
#
# Bottom-right corner:
# ────────────────────────────────────────────
#
# (x + bw, y + bh)
#
#
# Example with real values:
# ────────────────────────────────────────────
#
# x  = 100
# y  = 50
# bw = 200
# bh = 40
#
#
# This means:
# ────────────────────────────────────────────
#
# Box starts:
# - 100 pixels from the left
# - 50 pixels from the top
#
# Box size:
# - 200 pixels wide
# - 40 pixels tall
#
#
# Bottom-right calculation:
# ────────────────────────────────────────────
#
# x2 = x + bw
#    = 100 + 200
#    = 300
#
# y2 = y + bh
#    = 50 + 40
#    = 90
#
#
# Final box coordinates:
# ────────────────────────────────────────────
#
# Top-left:     (100, 50)
# Bottom-right: (300, 90)
#
#
# WHY NORMALIZE COORDINATES?
# ────────────────────────────────────────────
#
# Different devices have different screen sizes.
#
# Example:
#
# Phone width  = 1080 px
# Laptop width = 1920 px
#
# Raw pixel coordinates do not work consistently across devices.
#
# So we convert coordinates into values between 0 and 1.
#
#
# Normalization Formula:
# ────────────────────────────────────────────
#
# x1 = x / image_width
# y1 = y / image_height
#
# x2 = (x + bw) / image_width
# y2 = (y + bh) / image_height
#
#
# Example:
# ────────────────────────────────────────────
#
# Image width  = 1000
# Image height = 500
#
# x  = 100
# y  = 50
# bw = 200
# bh = 40
#
#
# Normalized coordinates:
# ────────────────────────────────────────────
#
# x1 = 100 / 1000 = 0.10
# y1 =  50 /  500 = 0.10
#
# x2 = 300 / 1000 = 0.30
# y2 =  90 /  500 = 0.18
#
#
# Final normalized bounding box:
# ────────────────────────────────────────────
#
# (0.10, 0.10, 0.30, 0.18)
#
#
# Meaning:
# ────────────────────────────────────────────
#
# The text occupies:
#
# - 10% → 30% of image width
# - 10% → 18% of image height
#
# Normalized coordinates work on ANY screen size.
# ────────────────────────────────────────────────
