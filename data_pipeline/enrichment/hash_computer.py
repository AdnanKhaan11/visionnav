"""
Hash computation for deduplication and integrity verification.

Two hash types:
  SHA-256: cryptographic, byte-identical detection
  pHash:   perceptual, near-duplicate detection

Computed during enrichment so they are available
before the deduplication gate runs.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from data_pipeline.core.exceptions import EnrichmentError
from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)


def enrich(sample: PipelineSample) -> None:
    """
    Compute and store image hashes on the sample.

    Modifies sample.enriched in place.
    No return value — caller checks sample.enriched after.

    Raises:
        EnrichmentError: if screenshot cannot be hashed
    """
    if sample.raw is None or not sample.raw.screenshot_path:
        return

    if sample.enriched is None:
        from data_pipeline.core.schemas import EnrichedContent

        sample.enriched = EnrichedContent()

    path = Path(sample.raw.screenshot_path)

    if not path.exists():
        raise EnrichmentError(
            message=f"Screenshot not found for hashing: {path}",
            sample_id=sample.sample_id,
            enricher="hash_computer",
            retryable=False,
        )

    # ── SHA-256 ────────────────────────────────────────────────────────────
    try:
        sample.enriched.image_hash_sha256 = _sha256(path)
    except Exception as exc:
        raise EnrichmentError(
            message=f"SHA-256 computation failed: {exc}",
            sample_id=sample.sample_id,
            enricher="hash_computer",
            retryable=True,
        )

    # ── pHash (best effort — not fatal if imagehash not installed) ─────────
    try:
        phash = _phash(path)
        if phash:
            sample.enriched.image_hash_phash = phash
    except Exception as exc:
        log.warning(
            "phash_failed",
            sample_id=sample.sample_id,
            error=str(exc),
        )
        # Not fatal — SHA-256 is sufficient for exact dedup


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _phash(path: Path) -> str | None:
    try:
        import imagehash
        from PIL import Image

        with Image.open(path) as img:
            return str(imagehash.phash(img.convert("RGB"), hash_size=16))
    except ImportError:
        return None
    except Exception:
        return None
