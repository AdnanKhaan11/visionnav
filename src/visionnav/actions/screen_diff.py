from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ChangeType(Enum):
    """Classification of how a screen changed."""

    NO_CHANGE = "no_change"
    MINOR_CHANGE = "minor_change"
    TEXT_UPDATE = "text_update"
    NEW_ELEMENT = "new_element"
    ELEMENT_REMOVED = "element_removed"
    LAYOUT_SHIFT = "layout_shift"
    FULL_SCREEN = "full_screen"


@dataclass(frozen=True)
class ScreenRegion:
    """A rectangular region of the screen in normalized coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    name: str

    def to_pixel_slice(self, width: int, height: int):
        """
        Convert normalized coordinates into numpy slice objects.

        Example:
            region = top_left
            returns:
                rows slice
                cols slice
        """

        r1 = int(self.y1 * height)
        r2 = int(self.y2 * height)

        c1 = int(self.x1 * width)
        c2 = int(self.x2 * width)

        return slice(r1, r2), slice(c1, c2)


@dataclass(frozen=True)
class DiffResult:
    """Complete result of comparing two screenshots."""

    change_type: ChangeType
    change_ratio: float
    changed_regions: list[str]
    most_changed_area: ScreenRegion
    is_significant: bool
    summary: str


class ScreenDiffAnalyzer:
    """
    Analyzes differences between two screenshots.

    Beyond simple pixel comparison:
    - WHERE on screen did change occur?
    - WHAT TYPE of change is it?
    - IS the change significant enough to indicate action succeeded?

    This gives the agent much richer feedback than a single number.
    """

    # Pixel intensity threshold used to determine
    # whether a pixel meaningfully changed.
    PIXEL_DIFF_THRESHOLD = 30

    # Region considered "active" if more than 5%
    # of its pixels changed.
    ACTIVE_REGION_THRESHOLD = 0.05

    # Nine regions dividing screen into 3x3 grid
    REGIONS: list[ScreenRegion] = [
        ScreenRegion(0.0, 0.0, 0.33, 0.33, "top_left"),
        ScreenRegion(0.33, 0.0, 0.67, 0.33, "top_center"),
        ScreenRegion(0.67, 0.0, 1.0, 0.33, "top_right"),
        ScreenRegion(0.0, 0.33, 0.33, 0.67, "middle_left"),
        ScreenRegion(0.33, 0.33, 0.67, 0.67, "center"),
        ScreenRegion(0.67, 0.33, 1.0, 0.67, "middle_right"),
        ScreenRegion(0.0, 0.67, 0.33, 1.0, "bottom_left"),
        ScreenRegion(0.33, 0.67, 0.67, 1.0, "bottom_center"),
        ScreenRegion(0.67, 0.67, 1.0, 1.0, "bottom_right"),
    ]

    def __init__(self, threshold: float = 0.01) -> None:
        self._threshold = threshold

    def analyze(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DiffResult:
        """
        Full analysis of screen change between two screenshots.

        Args:
            before:
                Screenshot before action.
                Shape: (H, W, 3)

            after:
                Screenshot after action.
                Shape: (H, W, 3)

        Returns:
            DiffResult object with complete analysis.
        """

        # -------------------------------------------------
        # Step 1: Validate shapes
        # -------------------------------------------------

        # If shapes differ, comparison becomes unreliable.
        # Safest assumption:
        # entire screen changed.
        if before.shape != after.shape:
            fallback_region = self.REGIONS[4]  # center

            return DiffResult(
                change_type=ChangeType.FULL_SCREEN,
                change_ratio=1.0,
                changed_regions=["entire_screen"],
                most_changed_area=fallback_region,
                is_significant=True,
                summary="Full screen change — screenshot shapes differ",
            )

        # -------------------------------------------------
        # Step 2: Compute diff map
        # -------------------------------------------------

        diff_map = self._compute_diff_map(before, after)

        # -------------------------------------------------
        # Step 3: Compute overall change ratio
        # -------------------------------------------------

        changed_pixels = diff_map > self.PIXEL_DIFF_THRESHOLD

        change_ratio = float(changed_pixels.mean())

        # -------------------------------------------------
        # Step 4: Analyze regions
        # -------------------------------------------------

        changed_regions, most_changed_area = self._analyze_regions(diff_map)

        # -------------------------------------------------
        # Step 5: Build region ratio dictionary
        # -------------------------------------------------

        region_ratios: dict[str, float] = {}

        height, width = diff_map.shape

        for region in self.REGIONS:

            rows, cols = region.to_pixel_slice(width, height)

            region_diff = diff_map[rows, cols]

            region_ratio = float((region_diff > self.PIXEL_DIFF_THRESHOLD).mean())

            region_ratios[region.name] = region_ratio

        # -------------------------------------------------
        # Step 6: Classify change
        # -------------------------------------------------

        change_type = self._classify_change(
            change_ratio,
            region_ratios,
        )

        # -------------------------------------------------
        # Step 7: Generate summary
        # -------------------------------------------------

        summary = self._generate_summary(
            change_type,
            changed_regions,
            change_ratio,
        )

        # -------------------------------------------------
        # Step 8: Build final result
        # -------------------------------------------------

        return DiffResult(
            change_type=change_type,
            change_ratio=change_ratio,
            changed_regions=changed_regions,
            most_changed_area=most_changed_area,
            is_significant=change_ratio >= self._threshold,
            summary=summary,
        )

    def _compute_diff_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-pixel change intensity.

        Returns:
            Float array of shape (H, W)

        Each pixel contains:
            average RGB channel difference
        """

        # Convert to float32 to prevent uint8 overflow
        before_f = before.astype(np.float32)
        after_f = after.astype(np.float32)

        # Absolute difference
        diff = np.abs(before_f - after_f)

        # Average RGB channels
        diff_map = diff.mean(axis=2)

        return diff_map

    def _analyze_regions(
        self,
        diff_map: np.ndarray,
    ) -> tuple[list[str], ScreenRegion]:
        """
        Analyze which regions changed most.

        Returns:
            changed_regions:
                region names sorted by change intensity

            most_changed_area:
                region with highest change ratio
        """

        height, width = diff_map.shape

        region_scores: list[tuple[ScreenRegion, float]] = []

        for region in self.REGIONS:

            # Extract region slice
            rows, cols = region.to_pixel_slice(width, height)

            region_diff = diff_map[rows, cols]

            # Fraction of significantly changed pixels
            change_ratio = float((region_diff > self.PIXEL_DIFF_THRESHOLD).mean())

            region_scores.append((region, change_ratio))

        # Sort highest change first
        region_scores.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        changed_regions = [region.name for region, ratio in region_scores if ratio > 0]

        most_changed_area = region_scores[0][0]

        return changed_regions, most_changed_area

    def _classify_change(
        self,
        change_ratio: float,
        region_ratios: dict[str, float],
    ) -> ChangeType:
        """
        Classify type of screen change.
        """

        # Count regions with meaningful activity
        active_regions = sum(
            ratio > self.ACTIVE_REGION_THRESHOLD for ratio in region_ratios.values()
        )

        # -----------------------------------------
        # Rule 1
        # -----------------------------------------

        if change_ratio < 0.001:
            return ChangeType.NO_CHANGE

        # -----------------------------------------
        # Rule 2
        # -----------------------------------------

        if change_ratio < 0.02:
            return ChangeType.MINOR_CHANGE

        # -----------------------------------------
        # Rule 3
        # -----------------------------------------

        if change_ratio > 0.50:
            return ChangeType.FULL_SCREEN

        # -----------------------------------------
        # Rule 4
        # -----------------------------------------

        if change_ratio > 0.15 and active_regions >= 5:
            return ChangeType.LAYOUT_SHIFT

        # -----------------------------------------
        # Rule 5
        # -----------------------------------------

        if 0.05 <= change_ratio <= 0.15 and active_regions <= 2:
            return ChangeType.NEW_ELEMENT

        # -----------------------------------------
        # Rule 6
        # -----------------------------------------

        if 0.02 <= change_ratio <= 0.05:
            return ChangeType.TEXT_UPDATE

        # -----------------------------------------
        # Safe fallback
        # -----------------------------------------

        return ChangeType.MINOR_CHANGE

    def _generate_summary(
        self,
        change_type: ChangeType,
        changed_regions: list[str],
        change_ratio: float,
    ) -> str:
        """
        Generate human-readable explanation.
        """

        top_regions = ", ".join(changed_regions[:3])

        if change_type == ChangeType.NO_CHANGE:
            return "No meaningful change detected"

        if change_type == ChangeType.MINOR_CHANGE:
            return (
                f"Minor change detected in {top_regions} " f"(ratio={change_ratio:.3f})"
            )

        if change_type == ChangeType.TEXT_UPDATE:
            return (
                f"Text update detected in {top_regions} " f"(ratio={change_ratio:.3f})"
            )

        if change_type == ChangeType.NEW_ELEMENT:
            return (
                f"New element appeared in {top_regions} " f"(ratio={change_ratio:.3f})"
            )

        if change_type == ChangeType.ELEMENT_REMOVED:
            return f"Element removed from {top_regions} " f"(ratio={change_ratio:.3f})"

        if change_type == ChangeType.LAYOUT_SHIFT:
            return (
                f"Major layout shift detected across "
                f"{top_regions} "
                f"(ratio={change_ratio:.3f})"
            )

        if change_type == ChangeType.FULL_SCREEN:
            return f"Full screen change detected " f"(ratio={change_ratio:.3f})"

        return "Unknown screen change detected"
