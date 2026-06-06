"""
Gate 4 — Hybrid Deduplication (Visual + Semantic Context).

Two-tier deduplication strategy for Vision-Language-Action datasets:

  Tier 1 — Exact:   SHA-256 byte match → always reject (byte-identical files).

  Tier 2 — Hybrid:  pHash distance ≤ PHASH_THRESHOLD → reject ONLY if semantic
                    context is also similar. Visual similarity alone is NOT
                    sufficient for rejection in a VLA dataset.

WHY HYBRID MATTERS:
  A VLA model receives (screenshot + task_instruction) as joint input.
  Two samples with visually identical screenshots but different task goals
  can require completely different actions — they are NOT training duplicates.

  Example rejected incorrectly by pHash-only:
    baf14295 step 4: type "Screenshots"  (task: create Screenshots in VisionNav_Data)
    89ad9eb3 step 4: type "Reports"      (task: create Reports in Work folder)
    pHash distance = 2  →  pHash-only would reject one
    Result: model loses the ability to learn task-conditional text entry

SEMANTIC DUPLICATE CONDITIONS (all must be true to reject):
  1. Action types match  (click ≠ type → keep both)
  2. Task instructions are similar  (Jaccard ≥ TASK_SIMILARITY_THRESHOLD)
  3. Target coordinates are in the same screen region  (click actions only)

INDEX FORMAT (version 2):
  {
    "version": 2,
    "sha256": {"hex": "sample_id"},
    "phash":  {"hex": {"sample_id": "...", "action_type": "...",
                       "task_words": "...", "coordinates": [x, y]}}
  }
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import structlog

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample

log = structlog.get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Maximum pHash Hamming distance to trigger semantic check.
# Can be higher than before because the semantic check prevents false positives.
# Images with distance > PHASH_THRESHOLD are always kept.
PHASH_THRESHOLD = 8

# Jaccard similarity threshold for task instruction comparison.
# If Jaccard(task_A_words, task_B_words) < this → different goals → keep both.
# 0.5 = 50% word overlap required to be considered "same goal".
TASK_SIMILARITY_THRESHOLD = 0.5

# Maximum normalized coordinate distance for click actions to be "same target".
# If the click target is > 15% of screen away → different target → keep both.
COORDINATE_TOLERANCE = 0.15

# Number of words from task instruction used for similarity comparison.
TASK_WORDS_LIMIT = 20

# Index file format version. Increment when phash value format changes.
_INDEX_VERSION = 2

# Action types that use coordinates for grounding.
_GROUNDING_ACTIONS = {"click", "double_click", "right_click"}


class Deduplicator:
    """
    Hybrid visual + semantic deduplicator for VLA training datasets.

    Usage:
        dedup = Deduplicator(index_path=Path("data/pipeline/.dedup_index.json"))
        dedup.validate(sample)   # raises ValidationError if duplicate
    """

    def __init__(self, index_path: Path) -> None:
        self._index_path   = Path(index_path)
        self._sha256_index: dict[str, str]  = {}   # hash → sample_id
        self._phash_index:  dict[str, dict] = {}   # hash → context_dict
        self._load_index()

    # ── Public API ──────────────────────────────────────────────────────────────

    def validate(self, sample: PipelineSample) -> None:
        """
        Check screenshot uniqueness against the index.
        Raises ValidationError (fatal=True) if a duplicate is detected.
        Adds sample to index if unique.
        """
        if sample.raw is None:
            return
        path = Path(sample.raw.screenshot_path)
        if not path.exists():
            return

        # ── Tier 1: SHA-256 exact duplicate ─────────────────────────────────
        sha256 = self._compute_sha256(path)
        if sha256 in self._sha256_index:
            original = self._sha256_index[sha256]
            raise ValidationError(
                message=(
                    f"Exact duplicate of {original} "
                    f"(SHA-256: {sha256[:16]}...). "
                    f"This screenshot has already been processed."
                ),
                sample_id = sample.sample_id,
                gate      = "dedup_sha256",
                is_fatal  = True,
            )

        # ── Tier 2: pHash near-duplicate with semantic context check ─────────
        phash = self._compute_phash(path)
        if phash is not None:
            duplicate = self._find_semantic_duplicate(sample, phash)
            if duplicate is not None:
                log.warning(
                    "near_duplicate_detected",
                    sample_id  = sample.sample_id,
                    similar_to = duplicate["sample_id"],
                    distance   = duplicate["distance"],
                )
                raise ValidationError(
                    message=(
                        f"Near-duplicate of sample {duplicate['sample_id']} "
                        f"(pHash distance={duplicate['distance']}, "
                        f"threshold={PHASH_THRESHOLD})"
                    ),
                    sample_id = sample.sample_id,
                    gate      = "dedup_phash",
                    is_fatal  = True,
                )

        # ── Not a duplicate: register and persist ───────────────────────────
        context = self._build_context(sample)
        self._sha256_index[sha256] = sample.sample_id
        if phash is not None:
            self._phash_index[phash] = context
        self._save_index()

        if sample.enriched:
            sample.enriched.image_hash_sha256 = sha256
            sample.enriched.image_hash_phash  = phash or ""
        if sample.quality:
            sample.quality.not_duplicate = True

    def index_size(self) -> dict[str, int]:
        return {
            "sha256": len(self._sha256_index),
            "phash":  len(self._phash_index),
        }

    # ── Core semantic duplicate check ───────────────────────────────────────────

    def _find_semantic_duplicate(
        self,
        sample: PipelineSample,
        new_phash: str,
    ) -> dict | None:
        """
        Scan pHash index for a sample that is both visually AND semantically
        similar to the new sample.

        Returns {"sample_id": ..., "distance": ...} if a true duplicate is found.
        Returns None if the sample should be kept.

        Visually similar samples are kept when they have:
          - Different action types (different skills)
          - Different task goals (different training scenarios)
          - Different target coordinates (different grounding targets)
        """
        for existing_hash, existing_ctx in self._phash_index.items():
            distance = self._hamming_distance(new_phash, existing_hash)
            if distance > PHASH_THRESHOLD:
                continue  # Visually different enough — keep without checking

            # Visual near-match found — check semantic context
            if self._is_semantic_duplicate(sample, existing_ctx):
                return {
                    "sample_id": existing_ctx.get("sample_id", "unknown"),
                    "distance":  distance,
                }

            # Visual near-match but semantically different — KEEP and log
            log.debug(
                "near_duplicate_kept",
                sample_id   = sample.sample_id,
                similar_to  = existing_ctx.get("sample_id"),
                distance    = distance,
                reason      = self._keep_reason(sample, existing_ctx),
            )

        return None

    def _is_semantic_duplicate(
        self,
        sample:       PipelineSample,
        existing_ctx: dict,
    ) -> bool:
        """
        Returns True only if ALL conditions hold — making this a true duplicate:
          1. Same action type
          2. Similar task instruction (Jaccard ≥ threshold)
          3. Same screen region (for grounding actions)

        If ANY condition fails → different training scenario → return False (keep).
        """
        if sample.raw is None:
            return True  # No context to compare — default conservative: reject

        new_action = sample.raw.action_type or ""
        ex_action  = existing_ctx.get("action_type", "")

        # ── Check 1: Action type must match ─────────────────────────────────
        # click vs type = completely different skills, always keep both
        if new_action != ex_action:
            return False

        # ── Check 2: Task goal similarity ───────────────────────────────────
        new_task_words = _task_word_set(sample.task or "")
        ex_task_words  = frozenset(
            (existing_ctx.get("task_words", "") or "").split()
        )
        task_sim = _jaccard(new_task_words, ex_task_words)
        if task_sim < TASK_SIMILARITY_THRESHOLD:
            # Different goals (e.g. "create Screenshots" vs "create Reports")
            # → different trajectories → keep both
            return False

        # ── Check 3: Screen region for grounding actions ────────────────────
        if new_action in _GROUNDING_ACTIONS:
            new_coords = sample.raw.coordinates if sample.raw else None
            ex_coords  = existing_ctx.get("coordinates")
            if not _same_screen_region(new_coords, ex_coords):
                # Different click targets → different grounding training signal
                return False

        # All checks passed — semantically the same scenario
        return True

    def _keep_reason(self, sample: PipelineSample, existing_ctx: dict) -> str:
        """Human-readable reason why a visually-similar sample was kept."""
        if sample.raw is None:
            return "no_raw"
        new_action = sample.raw.action_type or ""
        ex_action  = existing_ctx.get("action_type", "")
        if new_action != ex_action:
            return f"action_differs({new_action}≠{ex_action})"
        new_words = _task_word_set(sample.task or "")
        ex_words  = frozenset((existing_ctx.get("task_words", "") or "").split())
        sim       = _jaccard(new_words, ex_words)
        if sim < TASK_SIMILARITY_THRESHOLD:
            return f"task_differs(jaccard={sim:.2f})"
        if new_action in _GROUNDING_ACTIONS:
            new_c = sample.raw.coordinates
            ex_c  = existing_ctx.get("coordinates")
            if not _same_screen_region(new_c, ex_c):
                return f"coordinates_differ({new_c}≠{ex_c})"
        return "unknown"

    # ── Context helpers ─────────────────────────────────────────────────────────

    def _build_context(self, sample: PipelineSample) -> dict:
        """Build the metadata dict stored alongside a pHash entry."""
        action_type = ""
        coordinates = None
        if sample.raw:
            action_type = sample.raw.action_type or ""
            if action_type in _GROUNDING_ACTIONS:
                coordinates = sample.raw.coordinates

        task_words = " ".join(sorted(_task_word_set(sample.task or "")))

        return {
            "sample_id":   sample.sample_id,
            "action_type": action_type,
            "task_words":  task_words,
            "coordinates": coordinates,
        }

    # ── Hash computation ────────────────────────────────────────────────────────

    def _compute_sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _compute_phash(self, path: Path) -> str | None:
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
        try:
            import imagehash
            return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
        except Exception:
            if len(hash_a) != len(hash_b):
                return 999
            xor = int(hash_a, 16) ^ int(hash_b, 16)
            return bin(xor).count("1")

    # ── Index persistence ───────────────────────────────────────────────────────

    def _load_index(self) -> None:
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r") as f:
                data = json.load(f)

            found_version = data.get("version", 1)

            if found_version < _INDEX_VERSION:
                # Migrate: SHA-256 index is format-compatible; rebuild pHash index
                # from scratch on next pipeline run (pHash entries were string-only
                # in v1 and lacked context needed for semantic check).
                log.warning(
                    "dedup_index_migrating",
                    from_version = found_version,
                    to_version   = _INDEX_VERSION,
                    action       = "phash_index_cleared_will_rebuild",
                )
                self._sha256_index = data.get("sha256", {})
                self._phash_index  = {}
                self._save_index()
                log.info(
                    "dedup_index_loaded",
                    sha256_entries = len(self._sha256_index),
                    phash_entries  = 0,
                )
                return

            self._sha256_index = data.get("sha256", {})

            # Accept only new-format pHash entries (dict with sample_id key)
            self._phash_index = {
                k: v
                for k, v in data.get("phash", {}).items()
                if isinstance(v, dict) and "sample_id" in v
            }

            log.info(
                "dedup_index_loaded",
                sha256_entries = len(self._sha256_index),
                phash_entries  = len(self._phash_index),
            )
        except Exception as exc:
            log.warning("dedup_index_load_failed", error=str(exc))

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w") as f:
            json.dump(
                {
                    "version": _INDEX_VERSION,
                    "sha256":  self._sha256_index,
                    "phash":   self._phash_index,
                },
                f,
                indent=2,
            )


# ── Module-level pure functions (no class state needed) ───────────────────────

def _task_word_set(task: str) -> frozenset:
    """Extract the first TASK_WORDS_LIMIT words from a task string."""
    return frozenset(task.lower().split()[:TASK_WORDS_LIMIT])


def _jaccard(set_a: frozenset, set_b: frozenset) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|. Returns 1.0 if both empty."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _same_screen_region(
    coords_a: list | None,
    coords_b: list | None,
) -> bool:
    """
    True if both coordinate pairs are within COORDINATE_TOLERANCE of each other.
    Returns True (conservative) if either is None.
    """
    if coords_a is None or coords_b is None:
        return True
    if len(coords_a) < 2 or len(coords_b) < 2:
        return True
    dx = abs(float(coords_a[0]) - float(coords_b[0]))
    dy = abs(float(coords_a[1]) - float(coords_b[1]))
    return dx <= COORDINATE_TOLERANCE and dy <= COORDINATE_TOLERANCE