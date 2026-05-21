"""OCR Engine — PaddleOCR primary, Tesseract fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from visionnav.settings import OCRSettings

log = structlog.get_logger(__name__)


@dataclass
class TextRegion:
    text: str
    bbox: tuple[float, float, float, float]  # normalized (x1,y1,x2,y2)
    confidence: float


class OCREngine:
    """
    OCR pipeline for extracting text regions from screenshots.

    Architecture goals:
    - No hardcoded configuration
    - Configurable through OCRSettings
    - PaddleOCR primary with optional fallback
    - Structured TextRegion outputs
    """

    def __init__(
        self,
        settings: OCRSettings | None = None,
    ) -> None:
        # If no settings provided → create default OCRSettings
        self._settings = settings or OCRSettings()

        # Lazy-load PaddleOCR only when needed
        # Avoids heavy startup cost
        self._paddle = None

    def run(self, image: np.ndarray) -> list[TextRegion]:
        """
        Main OCR entrypoint.

        Engine behavior controlled entirely by settings:
        - paddle      → PaddleOCR only
        - tesseract   → Tesseract only
        - auto        → Paddle first, fallback to Tesseract
        """

        h, w = image.shape[:2]

        try:

            engine = self._settings.engine

            # Paddle only
            if engine == "paddle":

                regions = self._run_paddle(
                    image,
                    w,
                    h,
                )

            # Tesseract only
            elif engine == "tesseract":

                regions = self._run_tesseract(
                    image,
                    w,
                    h,
                )

            # Auto mode:
            # Try PaddleOCR first
            # If Paddle fails → fallback to Tesseract
            elif engine == "auto":

                try:

                    regions = self._run_paddle(
                        image,
                        w,
                        h,
                    )

                except Exception as exc:

                    log.warning(
                        "paddle_ocr_failed_falling_back_to_tesseract",
                        error=str(exc),
                    )

                    regions = self._run_tesseract(
                        image,
                        w,
                        h,
                    )

            else:

                log.warning(
                    "unknown_ocr_engine",
                    engine=engine,
                )

                return []

            # Apply max_regions ONCE globally
            return regions[: self._settings.max_regions]

        except Exception as exc:

            log.warning(
                "ocr_failed",
                error=str(exc),
            )

            return []

    def _run_paddle(
        self,
        image: np.ndarray,
        w: int,
        h: int,
    ) -> list[TextRegion]:
        from paddleocr import PaddleOCR

        # Lazy-load heavy OCR model only when needed
        if self._paddle is None:

            from paddleocr import PaddleOCR

            self._paddle = PaddleOCR(
                use_angle_cls=True,
                lang=self._settings.languages,
                show_log=False,
            )

        results = self._paddle.ocr(
            image,
            cls=True,
        )

        regions: list[TextRegion] = []

        # No OCR results found
        if not results or not results[0]:
            return regions

        for line in results[0]:

            pts, (text, conf) = line

            # Skip low-confidence detections
            if conf < self._settings.min_confidence:
                continue

            cleaned_text = text.strip()

            # Skip very short noisy text
            if len(cleaned_text) < self._settings.min_text_length:
                continue

            # Extract x/y coordinates separately
            # because Paddle returns 4 corner points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            regions.append(
                TextRegion(
                    text=cleaned_text,
                    bbox=(
                        min(xs) / w,
                        min(ys) / h,
                        max(xs) / w,
                        max(ys) / h,
                    ),
                    confidence=float(conf),
                )
            )

        return regions

    def _run_tesseract(
        self,
        image: np.ndarray,
        w: int,
        h: int,
    ) -> list[TextRegion]:

        import pytesseract
        from PIL import Image

        # Tesseract executable path from settings
        pytesseract.pytesseract.tesseract_cmd = self._settings.tesseract_path

        data = pytesseract.image_to_data(
            Image.fromarray(image),
            output_type=pytesseract.Output.DICT,
        )

        regions: list[TextRegion] = []

        for i, text in enumerate(data["text"]):

            cleaned_text = text.strip()

            # Tesseract confidence returned as 0-100
            conf = int(data["conf"][i])

            normalized_conf = conf / 100.0

            # Skip invalid detections
            if normalized_conf < self._settings.min_confidence or not cleaned_text:
                continue

            # Skip short noisy text
            if len(cleaned_text) < self._settings.min_text_length:
                continue

            x, y, bw, bh = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )

            regions.append(
                TextRegion(
                    text=cleaned_text,
                    bbox=(
                        x / w,
                        y / h,
                        (x + bw) / w,
                        (y + bh) / h,
                    ),
                    confidence=normalized_conf,
                )
            )

        # Additional noise filtering
        regions = [
            r
            for r in regions
            if not r.text.isnumeric()
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
