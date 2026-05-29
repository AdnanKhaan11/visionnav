"""
Gate 4 — Deduplication.
Detects exact and near-duplicate screenshots using hashing.

Two levels:
  SHA-256: byte-identical files (exact duplicates)
  pHash:   perceptually similar images (near-duplicates)

The hash index is persisted to disk so it survives process restarts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import structlog

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)

# Hamming distance threshold for pHash comparison
# 0 = identical pixels, 8 = visually similar (clock update, cursor move)
# 10 = noticeably different
PHASH_THRESHOLD = 8


class Deduplicator:
    """
    Maintains a hash index of seen screenshots.
    Detects duplicates at the sample level.

    Index structure (persisted as JSON):
    {
      "sha256": {"hash_value": "sample_id_first_seen"},
      "phash":  {"hash_value": "sample_id_first_seen"}
    }

    Usage:
        dedup = Deduplicator(index_path=Path("data/.dedup_index.json"))
        dedup.validate(sample)   # raises ValidationError if duplicate
    """

    def __init__(self, index_path: Path) -> None:
        self._index_path = Path(index_path)
        self._sha256_index: dict[str, str] = {}  # hash → sample_id
        self._phash_index: dict[str, str] = {}  # hash → sample_id
        self._load_index()

    def validate(self, sample: PipelineSample) -> None:
        """
        Check if this sample's screenshot already exists in the index.
        Raises ValidationError (fatal) if duplicate found.
        Adds to index if not a duplicate.
        """
        if sample.raw is None:
            return

        path = Path(sample.raw.screenshot_path)
        if not path.exists():
            return  # image validator already handles missing files

        # ── SHA-256 exact duplicate check ─────────────────────────────────
        sha256 = self._compute_sha256(path)
        if sha256 in self._sha256_index:
            original = self._sha256_index[sha256]
            raise ValidationError(
                message=(
                    f"Exact duplicate of sample {original}. "
                    f"SHA-256: {sha256[:16]}..."
                ),
                sample_id=sample.sample_id,
                gate="dedup_sha256",
                is_fatal=True,
            )

        # ── pHash near-duplicate check ────────────────────────────────────
        phash = self._compute_phash(path)
        if phash is not None:
            for existing_hash, existing_id in self._phash_index.items():
                distance = self._hamming_distance(phash, existing_hash)
                if distance <= PHASH_THRESHOLD:
                    log.warning(
                        "near_duplicate_detected",
                        sample_id=sample.sample_id,
                        similar_to=existing_id,
                        distance=distance,
                    )
                    raise ValidationError(
                        message=(
                            f"Near-duplicate of sample {existing_id} "
                            f"(pHash distance={distance}, threshold={PHASH_THRESHOLD})"
                        ),
                        sample_id=sample.sample_id,
                        gate="dedup_phash",
                        is_fatal=True,
                    )

        # Not a duplicate — add to index
        self._sha256_index[sha256] = sample.sample_id
        if phash is not None:
            self._phash_index[phash] = sample.sample_id
        self._save_index()

        # Store hashes in enriched content for later use
        if sample.enriched:
            sample.enriched.image_hash_sha256 = sha256
            sample.enriched.image_hash_phash = phash or ""

        if sample.quality:
            sample.quality.not_duplicate = True

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _compute_phash(self, path: Path) -> str | None:
        """
        Compute perceptual hash using imagehash library.
        Returns None if imagehash is not available.
        pHash is robust to minor visual changes (resizing, compression).
        """
        try:
            import imagehash
            from PIL import Image

            with Image.open(path) as img:
                return str(imagehash.phash(img.convert("RGB"), hash_size=16))
        except ImportError:
            log.warning("imagehash_not_available", action="skipping_phash_dedup")
            return None
        except Exception as exc:
            log.warning("phash_computation_failed", error=str(exc))
            return None

    def _hamming_distance(self, hash_a: str, hash_b: str) -> int:
        """
        Compute Hamming distance between two hex hash strings.
        Lower = more similar. 0 = identical.
        """
        try:
            import imagehash

            return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
        except Exception:
            # Fallback: bit-level comparison
            if len(hash_a) != len(hash_b):
                return 999
            a_int = int(hash_a, 16)
            b_int = int(hash_b, 16)
            xor = a_int ^ b_int
            return bin(xor).count("1")

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)
            self._sha256_index = data.get("sha256", {})
            self._phash_index = data.get("phash", {})
            log.info(
                "dedup_index_loaded",
                sha256_entries=len(self._sha256_index),
                phash_entries=len(self._phash_index),
            )
        except Exception as exc:
            log.warning("dedup_index_load_failed", error=str(exc))

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w") as f:
            json.dump(
                {"sha256": self._sha256_index, "phash": self._phash_index},
                f,
                indent=2,
            )

    def index_size(self) -> dict[str, int]:
        return {
            "sha256": len(self._sha256_index),
            "phash": len(self._phash_index),
        }
