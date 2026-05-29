"""
Gate 2 — Coordinate validation.
Coordinates must be normalized floats in [0.0, 1.0].
"""

from __future__ import annotations

from data_pipeline.core.exceptions import ValidationError
from data_pipeline.core.schemas import PipelineSample

# Only these action types have coordinates
_COORDINATE_ACTIONS = {"click", "double_click", "right_click"}


def validate(sample: PipelineSample) -> None:
    """
    Validate that click coordinates are in the valid [0.0, 1.0] range.
    Non-click actions with no coordinates pass automatically.
    Raises ValidationError (fatal) on any out-of-range coordinate.
    """
    if sample.raw is None:
        return  # schema validator already caught this

    action_type = sample.raw.action_type
    coords = sample.raw.coordinates

    # Non-click actions don't need coordinates
    if action_type not in _COORDINATE_ACTIONS:
        return

    # Click actions MUST have coordinates
    if coords is None:
        raise ValidationError(
            message=(
                f"Action type '{action_type}' requires coordinates " f"but none found"
            ),
            sample_id=sample.sample_id,
            gate="coordinates",
            is_fatal=True,
        )

    # Must be [x, y]
    if not isinstance(coords, (list, tuple)) or len(coords) != 2:
        raise ValidationError(
            message=(f"Coordinates must be [x, y], got: {coords!r}"),
            sample_id=sample.sample_id,
            gate="coordinates",
            is_fatal=True,
        )

    x, y = coords[0], coords[1]

    # Validate range
    for name, val in [("x", x), ("y", y)]:
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ValidationError(
                message=f"Coordinate {name} is not a number: {val!r}",
                sample_id=sample.sample_id,
                gate="coordinates",
                is_fatal=True,
            )

        if not (0.0 <= val <= 1.0):
            raise ValidationError(
                message=(
                    f"Coordinate {name}={val:.4f} is outside [0.0, 1.0]. "
                    f"This sample cannot produce valid click positions."
                ),
                sample_id=sample.sample_id,
                gate="coordinates",
                is_fatal=True,
            )

    # Update quality metrics
    if sample.quality:
        sample.quality.coordinates_valid = True
