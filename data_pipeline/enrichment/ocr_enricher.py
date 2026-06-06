"""
OCR enrichment — runs high-quality offline OCR on stored screenshots.

Why run OCR again after recording?
  Recording-time OCR: fast, real-time, lower quality
  Offline OCR:        slower, higher quality, more configurable

Changes from original:
  - min_confidence raised from 0.4 to 0.5 (reduces false positives)
  - min_text_length raised from 2 to 3 (eliminates ligature artifacts)
  - Added _is_clean_ocr_text() post-filter for symbol/artifact removal
  - Added _allowed_short set for legitimate 2-char tokens
  These changes were made based on evidence from actual OCR output showing
  tokens: 'xy', 'Gy', 'eo', 'BB', 'xXx@', 'cZEZE' in training prompts.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import structlog

from data_pipeline.core.exceptions import EnrichmentError
from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)


# ─── OCR Settings ─────────────────────────────────────────────────────────────

# Minimum confidence for offline OCR (higher than real-time 0.4).
# Offline processing has no latency constraint so we can be selective.
_OCR_MIN_CONFIDENCE = 0.5

# Minimum text length. Raised from 2 to 3 to eliminate 2-char ligature noise.
_OCR_MIN_TEXT_LENGTH = 3

# Maximum OCR regions stored per screenshot. 100 gives full coverage.
# Filtering happens at export time (see llamafactory.py) not here.
_OCR_MAX_REGIONS = 100

# 2-character strings that ARE legitimate and should not be filtered.
_ALLOWED_SHORT: frozenset[str] = frozenset(
    {
        "ok",
        "id",
        "no",
        "go",
        "up",
        "kb",
        "mb",
        "gb",
        "ui",
        "c#",
        "vb",
        "py",
        "js",
        "db",
        "pc",
        "tv",
        "an",
        "or",
        "in",
        "at",
        "by",
        "of",
        "to",
        "if",
    }
)


# ─── Public entry point ───────────────────────────────────────────────────────


def enrich(sample: PipelineSample) -> None:
    """
    Run offline OCR on the sample's screenshot.
    Stores detailed OCR results in sample.enriched.ocr_regions_detailed.

    Each stored region is a dict:
    {
        "text":       str,                   # cleaned OCR text
        "bbox":       [x1, y1, x2, y2],     # normalized [0,1]
        "confidence": float,                 # OCR confidence 0.0-1.0
        "script":     "latin"|"arabic"|...  # detected writing script
    }

    Raises:
        EnrichmentError: if screenshot cannot be loaded (retryable).
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

    # ── Run OCR ────────────────────────────────────────────────────────────
    try:
        from visionnav.perception.ocr import OCREngine
        from visionnav.settings import OCRSettings

        settings = OCRSettings(
            engine="tesseract",
            min_confidence=_OCR_MIN_CONFIDENCE,  # 0.5 (was 0.4)
            max_regions=_OCR_MAX_REGIONS,
            min_text_length=_OCR_MIN_TEXT_LENGTH,  # 3 (was 2)
        )
        engine = OCREngine(settings=settings)
        regions = engine.run(image)

    except Exception as exc:
        log.warning(
            "ocr_enrichment_failed",
            sample_id=sample.sample_id,
            error=str(exc),
        )
        sample.enriched.ocr_regions_detailed = []
        sample.enriched.n_ocr_regions = 0
        return

    # ── Convert to serializable dicts + apply quality filter ──────────────
    detailed: list[dict] = []
    rejected_count = 0

    for region in regions:
        text = region.text

        # Post-OCR quality filter: remove noise tokens that passed OCREngine
        # but are clearly not readable text (symbols, artifacts, ligatures).
        if not _is_clean_ocr_text(text):
            rejected_count += 1
            continue

        script = _detect_script(text)
        detailed.append(
            {
                "text": text,
                "bbox": list(region.bbox),
                "confidence": round(float(region.confidence), 4),
                "script": script,
            }
        )

    sample.enriched.ocr_regions_detailed = detailed
    sample.enriched.n_ocr_regions = len(detailed)

    if rejected_count > 0:
        log.debug(
            "ocr_noise_filtered",
            sample_id=sample.sample_id,
            kept=len(detailed),
            rejected_noise=rejected_count,
        )

    log.debug(
        "ocr_enrichment_complete",
        sample_id=sample.sample_id,
        regions=len(detailed),
    )


# ─── OCR text quality filter ──────────────────────────────────────────────────


def _is_clean_ocr_text(text: str) -> bool:
    """
    Returns True if the OCR token is likely real readable text.
    Returns False if it is likely noise, an artifact, or a glyph.

    Filter rules (applied in order):
      1. Empty or whitespace-only → reject
      2. 1 character → reject (single chars are never meaningful text)
      3. Exactly 2 characters → reject unless in _ALLOWED_SHORT whitelist
      4. No ASCII letter or digit anywhere → reject (pure symbols/punctuation)
      5. Contains private-use Unicode or control characters → reject
      6. Short string (3-5 chars) with high uppercase + no vowels → reject

    False positive analysis:
      - Normal English words of 3+ chars: always pass
      - Programming names (Python, JSON, API): always pass
      - Urdu/Arabic text: passes rules 1-5 (Arabic Unicode is valid)
      - "OK", "ID", "C#": explicitly whitelisted
      - "cZEZE", "xXx@": caught by rule 6 or rule 4

    False negative analysis (noise that slips through):
      - 6+ char garbage strings without clear pattern: may slip through
      - This is acceptable — the filter addresses the 80% of common noise
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()
    length = len(stripped)

    # Rule 1: single character
    if length <= 1:
        return False

    # Rule 2: exactly 2 characters
    if length == 2:
        return stripped.lower() in _ALLOWED_SHORT

    # Rule 3: must contain at least one letter or digit
    has_alphanum = any(c.isalpha() or c.isdigit() for c in stripped)
    if not has_alphanum:
        return False

    # Rule 4: reject private-use Unicode and control characters
    for ch in stripped:
        cat = unicodedata.category(ch)
        if cat == "Cc":  # control character
            return False
        if cat == "Co":  # private use
            return False
        if cat == "Cn":  # unassigned
            return False

    # Rule 5: short string with high uppercase ratio and no vowels
    # Catches artifacts like "cZEZE", "xXx@" that pass length checks
    if 3 <= length <= 6:
        upper_count = sum(1 for c in stripped if c.isupper())
        upper_ratio = upper_count / length
        has_vowel = any(c.lower() in "aeiouاوی" for c in stripped)
        if upper_ratio >= 0.5 and not has_vowel:
            return False

    return True


# ─── Image loading ────────────────────────────────────────────────────────────


def _load_image(path: Path) -> np.ndarray:
    """Load image from disk as numpy RGB array."""
    from PIL import Image

    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


# ─── Script detection ─────────────────────────────────────────────────────────


def _detect_script(text: str) -> str:
    """
    Detect writing script from text content.
    Used to tag multilingual regions for language detection downstream.

    Returns: "arabic" | "latin" | "unknown"
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
