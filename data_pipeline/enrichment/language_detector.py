"""
Language detection from OCR text content.

Detects the primary language displayed on the screen.
Used to tag multilingual samples and enable language-specific processing.

Detection strategy:
  1. Script detection (Arabic script → Urdu or Pashto)
  2. Word frequency analysis (common words per language)
  3. Character n-gram patterns

For Stage 1: simple rule-based detection (fast, no model needed)
For Stage 3+: replace with fasttext language detection model
"""

from __future__ import annotations

from data_pipeline.core.schemas import Language, PipelineSample

import structlog

log = structlog.get_logger(__name__)

# Common words per language for word-frequency detection
_LANGUAGE_MARKERS: dict[str, set[str]] = {
    Language.URDU.value: {
        "کھولیں",
        "بند",
        "ارسال",
        "ای",
        "میل",
        "کریں",
        "نیا",
        "فائل",
        "ترتیبات",
        "پاس",
        "ورڈ",
        "لاگ",
        "ان",
        "آؤٹ",
        "تلاش",
        "محفوظ",
    },
    Language.PASHTO.value: {
        "خلاص",
        "وتل",
        "لیدل",
        "کول",
        "لیږل",
        "ټولګه",
        "پاسوورډ",
    },
    Language.ARABIC.value: {
        "افتح",
        "أغلق",
        "إرسال",
        "بريد",
        "ملف",
        "جديد",
        "حفظ",
        "بحث",
    },
    Language.HINDI.value: {
        "खोलें",
        "बंद",
        "भेजें",
        "फ़ाइल",
        "नया",
        "सहेजें",
        "खोजें",
    },
}


def enrich(sample: PipelineSample) -> None:
    """
    Detect the primary language of the screen from OCR regions.
    Sets sample.enriched.detected_language.

    Detection is best-effort — defaults to UNKNOWN if unsure.
    Never raises — language detection failure is non-critical.
    """
    if sample.enriched is None:
        return

    # Collect all OCR text
    ocr_text = _collect_text(sample)

    if not ocr_text.strip():
        sample.enriched.detected_language = Language.UNKNOWN.value
        return

    try:
        detected = _detect(ocr_text)
        sample.enriched.detected_language = detected

        if detected != Language.ENGLISH.value:
            # Tag multilingual samples for easy filtering
            if "multilingual" not in sample.tags:
                sample.tags.append("multilingual")
            if detected not in sample.tags:
                sample.tags.append(f"lang:{detected}")

        log.debug(
            "language_detected",
            sample_id=sample.sample_id,
            language=detected,
        )

    except Exception as exc:
        log.warning(
            "language_detection_failed",
            sample_id=sample.sample_id,
            error=str(exc),
        )
        sample.enriched.detected_language = Language.UNKNOWN.value


def _collect_text(sample: PipelineSample) -> str:
    """
    Collect all text from enriched OCR regions.
    Falls back to raw OCR text if enrichment not done yet.
    """
    if sample.enriched and sample.enriched.ocr_regions_detailed:
        return " ".join(r["text"] for r in sample.enriched.ocr_regions_detailed)

    if sample.raw and sample.raw.ocr_text_raw:
        return sample.raw.ocr_text_raw

    return ""


def _detect(text: str) -> str:
    """
    Detect language from text using script + word frequency.

    Returns Language enum value string.
    """
    # ── Step 1: Arabic script check ────────────────────────────────────────
    arabic_char_ratio = sum(
        1 for ch in text if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f"
    ) / max(len(text), 1)

    if arabic_char_ratio > 0.2:
        # Distinguish Urdu vs Pashto vs Arabic by word markers
        text_words = set(text.split())

        urdu_score = len(text_words & _LANGUAGE_MARKERS[Language.URDU.value])
        pashto_score = len(text_words & _LANGUAGE_MARKERS[Language.PASHTO.value])
        arabic_score = len(text_words & _LANGUAGE_MARKERS[Language.ARABIC.value])

        if urdu_score >= pashto_score and urdu_score >= arabic_score:
            return Language.URDU.value
        elif pashto_score > urdu_score:
            return Language.PASHTO.value
        elif arabic_score > 0:
            return Language.ARABIC.value
        else:
            return Language.URDU.value  # default for Arabic-script content

    # ── Step 2: Devanagari script (Hindi) ─────────────────────────────────
    devanagari_ratio = sum(1 for ch in text if "\u0900" <= ch <= "\u097f") / max(
        len(text), 1
    )

    if devanagari_ratio > 0.2:
        return Language.HINDI.value

    # ── Step 3: Default to English ─────────────────────────────────────────
    latin_ratio = sum(1 for ch in text if "a" <= ch.lower() <= "z") / max(len(text), 1)

    if latin_ratio > 0.3:
        return Language.ENGLISH.value

    return Language.UNKNOWN.value
