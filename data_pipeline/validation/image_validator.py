"""
Gate 3 — Image quality validation.
Screenshot must exist, be readable, and meet minimum quality standards.
"""

from __future__ import annotations

from pathlib import Path

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample

# Minimum acceptable image dimensions
MIN_WIDTH = 400  # pixels
MIN_HEIGHT = 300  # pixels

# Minimum file size — below this it's likely blank or corrupted
MIN_FILE_SIZE_KB = 10.0


def validate(sample: PipelineSample) -> None:
    """
    Validate screenshot file existence and basic image quality.
    Raises ValidationError (non-fatal) on quality failures.
    Non-fatal because the recording may be usable with manual review.
    """
    if sample.raw is None:
        return

    path = Path(sample.raw.screenshot_path)

    # ── File must exist ───────────────────────────────────────────────────
    if not path.exists():
        raise ValidationError(
            message=f"Screenshot not found: {path}",
            sample_id=sample.sample_id,
            gate="image",
            is_fatal=False,  # might be on different machine (path issue)
        )

    # ── File must have minimum size ───────────────────────────────────────
    size_kb = path.stat().st_size / 1024
    if size_kb < MIN_FILE_SIZE_KB:
        raise ValidationError(
            message=(
                f"Screenshot too small: {size_kb:.1f}KB "
                f"(minimum {MIN_FILE_SIZE_KB}KB). "
                f"Likely blank or corrupted."
            ),
            sample_id=sample.sample_id,
            gate="image",
            is_fatal=False,
        )

    # ── File must be a readable image ─────────────────────────────────────
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format
    except Exception as exc:
        raise ValidationError(
            message=f"Cannot read screenshot: {exc}",
            sample_id=sample.sample_id,
            gate="image",
            is_fatal=False,
        )

    # ── Minimum resolution ─────────────────────────────────────────────────
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise ValidationError(
            message=(
                f"Screenshot resolution {width}x{height} below minimum "
                f"{MIN_WIDTH}x{MIN_HEIGHT}. Too small for reliable OCR."
            ),
            sample_id=sample.sample_id,
            gate="image",
            is_fatal=False,
        )

    # ── Check for blank screens ────────────────────────────────────────────
    # A blank screen has very low pixel variance
    try:
        import numpy as np
        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            arr = np.array(img.convert("RGB"))
            variance = float(arr.var())

        if variance < 50.0:
            raise ValidationError(
                message=(
                    f"Screenshot appears blank (pixel variance={variance:.1f}). "
                    f"Likely a failed capture."
                ),
                sample_id=sample.sample_id,
                gate="image",
                is_fatal=False,
            )
    except ImportError:
        pass  # numpy not available → skip variance check

    # Passed — update quality metrics
    if sample.quality:
        sample.quality.image_valid = True
