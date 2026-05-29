"""
OCR enrichment — runs high-quality offline OCR on stored screenshots.

Why run OCR again after recording?
  Recording-time OCR: fast, real-time, lower quality
  Offline OCR:        slow, batch, higher quality + more detail

The offline pass produces richer OCR data:
  - More text regions detected
  - Higher confidence scores
  - Better bounding box precision
  - Full metadata per region

This richer data improves:
  - Annotation quality (more context for reasoning generation)
  - Model training (more accurate element positions)
  - Reasoning verification (can verify claims against OCR)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from data_pipeline.core.exceptions import EnrichmentError
from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)


def enrich(sample: PipelineSample) -> None:
    """
    Run offline OCR on the sample's screenshot.
    Stores detailed results in sample.enriched.ocr_regions_detailed.

    Each region in the detailed list is a dict:
    {
        "text":       str,
        "bbox":       [x1, y1, x2, y2],   # normalized [0,1]
        "confidence": float,
        "script":     "latin" | "arabic" | "unknown"
    }
    """
    if sample.raw is None or not sample.raw.screenshot_path:
        return

    if sample.enriched is None:
        from data_pipeline.core.schemas import EnrichedContent

        sample.enriched = EnrichedContent()

    path = Path(sample.raw.screenshot_path)
    if not path.exists():
        raise EnrichmentError(
            message=f"Screenshot not found for OCR: {path}",
            sample_id=sample.sample_id,
            enricher="ocr_enricher",
            retryable=False,
        )

    # ── Load image ─────────────────────────────────────────────────────────
    try:
        image = _load_image(path)
    except Exception as exc:
        raise EnrichmentError(
            message=f"Cannot load screenshot for OCR: {exc}",
            sample_id=sample.sample_id,
            enricher="ocr_enricher",
            retryable=True,
        )

    h, w = image.shape[:2]
    sample.enriched.image_width = w
    sample.enriched.image_height = h

    # ── Run OCR using existing VisionNav OCR engine ────────────────────────
    try:
        from visionnav.perception.ocr import OCREngine, TextRegion
        from visionnav.settings import OCRSettings

        # Use Tesseract with slightly lower confidence threshold
        # for offline pass — we want more regions, filter later
        settings = OCRSettings(
            engine="tesseract",
            min_confidence=0.4,  # lower than real-time 0.6 → more regions
            max_regions=100,
            min_text_length=2,
        )
        engine = OCREngine(settings=settings)
        regions = engine.run(image)

    except Exception as exc:
        log.warning(
            "ocr_enrichment_failed",
            sample_id=sample.sample_id,
            error=str(exc),
        )
        # Not fatal — sample can still be trained on without enriched OCR
        sample.enriched.ocr_regions_detailed = []
        sample.enriched.n_ocr_regions = 0
        return

    # ── Convert TextRegion list to serializable dicts ──────────────────────
    detailed: list[dict] = []
    for region in regions:
        script = _detect_script(region.text)
        detailed.append(
            {
                "text": region.text,
                "bbox": list(region.bbox),
                "confidence": round(float(region.confidence), 4),
                "script": script,
            }
        )

    sample.enriched.ocr_regions_detailed = detailed
    sample.enriched.n_ocr_regions = len(detailed)

    log.debug(
        "ocr_enrichment_complete",
        sample_id=sample.sample_id,
        regions=len(detailed),
    )


def _load_image(path: Path) -> np.ndarray:
    """Load image from disk as numpy RGB array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _detect_script(text: str) -> str:
    """
    Detect writing script from text content.
    Needed to distinguish Latin (English) from Arabic script (Urdu/Pashto).
    """
    if not text:
        return "unknown"

    arabic_count = sum(
        1
        for ch in text
        if "\u0600" <= ch <= "\u06ff"  # Arabic Unicode block
        or "\u0750" <= ch <= "\u077f"  # Arabic Supplement
        or "\ufb50" <= ch <= "\ufdff"  # Arabic Presentation Forms A
        or "\ufe70" <= ch <= "\ufeff"  # Arabic Presentation Forms B
    )

    if arabic_count / max(len(text), 1) > 0.3:
        return "arabic"

    latin_count = sum(
        1 for ch in text if ("a" <= ch.lower() <= "z") or ch in "0123456789"
    )

    if latin_count / max(len(text), 1) > 0.3:
        return "latin"

    return "unknown"
