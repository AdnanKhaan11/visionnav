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
        self._paddle = None  # lazy-loaded

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
        if not results or not results[0]:
            return regions
        for line in results[0]:
            pts, (text, conf) = line
            if conf < self._min_confidence:
                continue
            xs = [p[0] for p in pts]
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
        return regions
