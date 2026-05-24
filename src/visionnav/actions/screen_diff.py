"""
ScreenDiffAnalyzer — detects and classifies changes between two screenshots.

The agent uses this to answer three questions after every action:
  1. Did anything change?      → change_ratio
  2. Where did it change?      → changed_regions, most_changed_area
  3. What type of change?      → change_type (new window? text? full screen?)

This is richer feedback than a single True/False from the old verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np

# ─── Change Classification ────────────────────────────────────────────────────


class ChangeType(Enum):
    """
    What kind of screen change occurred.

    Ordered from least to most significant.
    The classifier checks rules in this priority order.
    """

    NO_CHANGE = "no_change"  # nothing changed
    MINOR_CHANGE = "minor_change"  # cursor blink, clock tick
    TEXT_UPDATE = "text_update"  # text value changed in a field
    NEW_ELEMENT = "new_element"  # small new UI element appeared
    ELEMENT_REMOVED = "element_removed"  # reserved — visually same as NEW_ELEMENT
    LAYOUT_SHIFT = "layout_shift"  # dialog or new window appeared
    FULL_SCREEN = "full_screen"  # entire screen replaced (app switch)


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScreenRegion:
    """
    A named rectangle on the screen in normalized coordinates.

    All values are between 0.0 and 1.0:
      (0.0, 0.0) = top-left corner
      (1.0, 1.0) = bottom-right corner

    frozen=True means this object cannot be modified after creation.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    name: str

    def to_pixel_slice(
        self,
        height: int,
        width: int,
    ) -> tuple[slice, slice]:
        """
        Convert normalized(between 0.0 and 1.0) coordinates to numpy array slices.

        Args:
            height: screen height in pixels (first dimension in numpy dimentioon means number of rows)
            width:  screen width  in pixels (second dimension in numpy dimention means number of columns)

        Returns:
            (row_slice, col_slice) — use as: array[row_slice, col_slice] --> why we return slices?

        Example:
            region = ScreenRegion(0.0, 0.0, 0.5, 0.5, "top_left")
            rows, cols = region.to_pixel_slice(1080, 1920)
            sub_image = diff_map[rows, cols]  # top-left quarter
        """
        row_start = int(self.y1 * height)
        row_end = int(self.y2 * height)
        col_start = int(self.x1 * width)
        col_end = int(self.x2 * width)
        return slice(row_start, row_end), slice(col_start, col_end)


@dataclass(frozen=True)
class DiffResult:
    """
    Complete result of comparing two screenshots.

    All fields are read-only after creation.

    quality_interpretation:
        change_ratio < 0.001  → action had no effect
        change_ratio < 0.02   → minor noise, probably no real change
        change_ratio > 0.50   → major change, new screen or app
    """

    change_type: ChangeType
    change_ratio: float  # 0.0 to 1.0 — fraction of pixels changed
    changed_regions: tuple[str, ...]  # region names sorted by change intensity
    most_changed_area: ScreenRegion  # region with highest change
    is_significant: bool  # True if change_ratio >= threshold
    summary: str  # human-readable description


# ─── Analyzer ─────────────────────────────────────────────────────────────────


class ScreenDiffAnalyzer:
    """
    Compares two screenshots and classifies how the screen changed.

    Usage:
        analyzer = ScreenDiffAnalyzer()
        result   = analyzer.analyze(before_screenshot, after_screenshot)

        print(result.change_type)    # ChangeType.NEW_ELEMENT
        print(result.summary)        # "New element appeared in center"
        print(result.is_significant) # True
    """

    # ── Constants ─────────────────────────────────────────────────────────────

    # A pixel is "changed" if any channel differs by more than this.
    # Threshold of 30 filters out anti-aliasing noise and clock updates.
    # Real UI changes differ by 50–255 intensity units.
    PIXEL_THRESHOLD: ClassVar[int] = 30

    # A region is "active" if more than 5% of its pixels changed.
    # Below this — probably just noise within the region.
    ACTIVE_REGION_THRESHOLD: ClassVar[float] = 0.05

    # Screen divided into 3x3 grid — nine named regions.
    # Lets us say "change in top_right" instead of raw coordinates.
    REGIONS: ClassVar[list[ScreenRegion]] = [
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
        """
        Args:
            threshold: minimum change_ratio to mark result as significant.
                       0.01 = at least 1% of pixels must change.
        """
        self._threshold = threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DiffResult:
        """
        Compare two screenshots and return a complete change analysis.

        Args:
            before: screenshot before the action — shape (H, W, 3) uint8
            after:  screenshot after  the action — shape (H, W, 3) uint8

        Returns:
            DiffResult with change_type, changed_regions, summary, etc.
        """
        # ── Guard: shapes must match ──────────────────────────────────────
        # If shapes differ we cannot compare pixel by pixel.
        # Return FULL_SCREEN as a safe pessimistic assumption.
        # But If Sizes Differ...
        # Eventually one image has pixels the other image DOES NOT HAVE.so its not match perfectly
        if before.shape != after.shape:
            return DiffResult(
                change_type=ChangeType.FULL_SCREEN,
                change_ratio=1.0,
                changed_regions=("entire_screen",),
                most_changed_area=self.REGIONS[4],  # center region
                is_significant=True,
                summary="Full screen change — screenshot sizes differ",
            )

        # ── Step 1: pixel-level diff map ──────────────────────────────────
        diff_map = self._compute_diff_map(before, after)

        # ── Step 2: overall change ratio ──────────────────────────────────
        change_ratio = float((diff_map > self.PIXEL_THRESHOLD).mean())

        # ── Step 3: per-region analysis ───────────────────────────────────
        # Compute once and reuse — avoids computing region ratios twice.
        region_ratios = self._compute_region_ratios(diff_map)

        # ── Step 4: sort regions by change intensity ───────────────────────
        # we are sorting this because we want to know which regions changed the most, so we can say "the center changed a lot but the edges didn't"
        sorted_regions = sorted(
            region_ratios.items(),  # What does .items() do? It converts the dictionary into a list of pairs: example : {"top_left": 0.02, "center": 0.45} --> [("top_left", 0.02), ("center", 0.45)]
            key=lambda item: item[1],
            reverse=True,
        )
        # This line picks only the region names where something actually changed.
        # example :  changed_regions = ("center", "top_center", "middle_right", "top_left", ...)
        changed_regions = tuple(name for name, ratio in sorted_regions if ratio > 0)
        most_changed_name = sorted_regions[0][0]

        #         We have the name "center" — but we need the full ScreenRegion object (with x1, y1, x2, y2 coordinates).
        #          self.REGIONS is our list of 9 ScreenRegion objects. We search through them:
        most_changed_area = next(r for r in self.REGIONS if r.name == most_changed_name)

        # ── Step 5: classify ──────────────────────────────────────────────
        change_type = self._classify_change(change_ratio, region_ratios)

        # ── Step 6: summarize ─────────────────────────────────────────────
        summary = self._generate_summary(
            change_type, list(changed_regions), change_ratio
        )

        return DiffResult(
            change_type=change_type,
            change_ratio=change_ratio,
            changed_regions=changed_regions,
            most_changed_area=most_changed_area,
            is_significant=change_ratio >= self._threshold,
            summary=summary,
        )

    # ── Private Methods ───────────────────────────────────────────────────────

    def _compute_diff_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-pixel change intensity as a 2D map.

        Why float32?
        uint8 subtraction wraps around: 50 - 200 = 106 (wrong).
        float32 gives the correct result: 50 - 200 = -150, abs = 150.

        Why mean across channels?
        Reduces 3D (H, W, 3) → 2D (H, W).
        Each pixel becomes one number: "how much did this pixel change overall?"

        Returns:
            Array shape (H, W) — each value is average channel change (0–255).
        """
        before_f = before.astype(np.float32)
        after_f = after.astype(np.float32)
        diff = np.abs(before_f - after_f)
        return diff.mean(axis=2)

    def _compute_region_ratios(
        self,
        diff_map: np.ndarray,
    ) -> dict[str, float]:
        """
        For each of the 9 screen regions, compute what fraction of
        its pixels changed significantly.

        Returns:
            {"top_left": 0.02, "center": 0.45, ...}
        """
        height, width = diff_map.shape
        ratios: dict[str, float] = {}

        for region in self.REGIONS:
            rows, cols = region.to_pixel_slice(height, width)
            region_diff = diff_map[rows, cols]
            ratio = float((region_diff > self.PIXEL_THRESHOLD).mean())
            ratios[region.name] = ratio

        return ratios

    def _classify_change(
        self,
        change_ratio: float,
        region_ratios: dict[str, float],
    ) -> ChangeType:
        """
        Classify the type of screen change using priority rules.

        Rules are checked in order — first match wins.
        Most obvious changes (NO_CHANGE, FULL_SCREEN) checked first.
        Subtle changes (TEXT_UPDATE) checked last.

        active_regions = regions where more than 5% of pixels changed.
        Spread of active regions tells us whether it's a dialog (few regions)
        or a full layout change (many regions).
        """
        active_regions = sum(
            1
            for ratio in region_ratios.values()
            if ratio > self.ACTIVE_REGION_THRESHOLD
        )

        # Rule 1 — nothing changed (noise level)
        if change_ratio < 0.001:
            return ChangeType.NO_CHANGE

        # Rule 2 — tiny change (cursor blink, clock update)
        if change_ratio < 0.02:
            return ChangeType.MINOR_CHANGE

        # Rule 3 — more than half the screen changed (app switch)
        if change_ratio > 0.50:
            return ChangeType.FULL_SCREEN

        # Rule 4 — large change spread across many regions (new window/dialog)
        if change_ratio > 0.15 and active_regions >= 5:
            return ChangeType.LAYOUT_SHIFT

        # Rule 5 — medium change concentrated in 1-2 regions (new button/icon)
        if 0.05 <= change_ratio <= 0.15 and active_regions <= 2:
            return ChangeType.NEW_ELEMENT

        # Rule 6 — small but real change (text typed, value changed)
        if 0.02 <= change_ratio <= 0.05:
            return ChangeType.TEXT_UPDATE

        # Default — something changed but doesn't fit above categories
        return ChangeType.MINOR_CHANGE

    def _generate_summary(
        self,
        change_type: ChangeType,
        changed_regions: list[str],
        change_ratio: float,
    ) -> str:
        """
        Generate a plain-English summary of what changed.

        Uses top 3 changed regions to keep summaries concise.
        """
        top = ", ".join(changed_regions[:3]) if changed_regions else "unknown area"
        pct = f"{change_ratio * 100:.1f}%"

        summaries: dict[ChangeType, str] = {
            ChangeType.NO_CHANGE: "No meaningful change detected",
            ChangeType.MINOR_CHANGE: f"Minor change in {top} — likely cursor or clock ({pct} of pixels)",
            ChangeType.TEXT_UPDATE: f"Text updated in {top} ({pct} of pixels changed)",
            ChangeType.NEW_ELEMENT: f"New element appeared in {top} ({pct} of pixels changed)",
            ChangeType.ELEMENT_REMOVED: f"Element removed from {top} ({pct} of pixels changed)",
            ChangeType.LAYOUT_SHIFT: f"Major layout change across {top} — possible new window ({pct})",
            ChangeType.FULL_SCREEN: f"Full screen change — application likely switched ({pct})",
        }

        return summaries.get(change_type, "Unknown screen change detected")
