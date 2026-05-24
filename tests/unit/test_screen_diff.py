import numpy as np
import pytest

from visionnav.actions.screen_diff import (
    ChangeType,
    ScreenDiffAnalyzer,
)


# Use this helper to create test images
def make_screen(
    height=100,
    width=200,
    color=(255, 255, 255),
) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)


def paint_region(
    screen,
    y1_ratio,
    x1_ratio,
    y2_ratio,
    x2_ratio,
    color,
):
    """Paint a rectangular area on a screen array."""

    h, w = screen.shape[:2]

    r1 = int(y1_ratio * h)
    r2 = int(y2_ratio * h)

    c1 = int(x1_ratio * w)
    c2 = int(x2_ratio * w)

    screen[r1:r2, c1:c2] = color

    return screen


# ---------------------------------------------------------
# Test 1: Identical screens → NO_CHANGE
# ---------------------------------------------------------


def test_identical_screens():
    screen = make_screen()

    analyzer = ScreenDiffAnalyzer()

    result = analyzer.analyze(screen, screen.copy())

    assert result.change_type == ChangeType.NO_CHANGE
    assert result.change_ratio == 0.0
    assert result.is_significant is False


# ---------------------------------------------------------
# Test 2: Completely different screens → FULL_SCREEN
# ---------------------------------------------------------


def test_full_screen_change():
    before = make_screen(color=(255, 255, 255))
    after = make_screen(color=(0, 0, 0))

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert result.change_type == ChangeType.FULL_SCREEN
    assert result.change_ratio > 0.9


# ---------------------------------------------------------
# Test 3: Small change in one region → NEW_ELEMENT
# ---------------------------------------------------------


def test_new_element_detected():
    before = make_screen()
    after = make_screen()

    # Paint black rectangle in center
    after = paint_region(
        after,
        0.4,
        0.4,
        0.6,
        0.6,
        (0, 0, 0),
    )

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert result.change_type in (
        ChangeType.NEW_ELEMENT,
        ChangeType.TEXT_UPDATE,
    )

    assert "center" in result.changed_regions


# ---------------------------------------------------------
# Test 4: Very tiny change → MINOR_CHANGE
# ---------------------------------------------------------


def test_minor_change():
    before = make_screen()
    after = make_screen()

    # Simulate tiny cursor blink
    after[5][5] = [200, 200, 200]
    after[5][6] = [200, 200, 200]
    after[5][7] = [200, 200, 200]

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert result.change_type in (
        ChangeType.NO_CHANGE,
        ChangeType.MINOR_CHANGE,
    )


# ---------------------------------------------------------
# Test 5: most_changed_area is correct region
# ---------------------------------------------------------


def test_most_changed_area_region():
    before = make_screen()
    after = make_screen()

    # Paint bottom_right heavily
    after = paint_region(
        after,
        0.7,
        0.7,
        1.0,
        1.0,
        (0, 0, 0),
    )

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert result.most_changed_area.name == "bottom_right"


# ---------------------------------------------------------
# Test 6: DiffResult is immutable
# ---------------------------------------------------------


def test_diff_result_immutable():
    before = make_screen()
    after = make_screen(color=(0, 0, 0))

    result = ScreenDiffAnalyzer().analyze(before, after)

    with pytest.raises((AttributeError, TypeError)):
        result.change_ratio = 0.0


# ---------------------------------------------------------
# Test 7: Different shapes handled gracefully
# ---------------------------------------------------------


def test_different_shape_handled():
    before = make_screen(height=100, width=200)

    after = make_screen(
        height=200,
        width=400,
    )

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert result.change_type == ChangeType.FULL_SCREEN


# ---------------------------------------------------------
# Test 8: Summary is human-readable
# ---------------------------------------------------------


def test_summary_is_string():
    before = make_screen()
    after = make_screen(color=(0, 0, 0))

    result = ScreenDiffAnalyzer().analyze(before, after)

    assert isinstance(result.summary, str)
    assert len(result.summary) > 5


# ---------------------------------------------------------
# Test 9: Regions sorted correctly
# ---------------------------------------------------------


def test_changed_regions_sorted():
    before = make_screen(height=300, width=300)
    after = make_screen(height=300, width=300)

    # Heavy change in bottom_right
    after = paint_region(
        after,
        0.67,
        0.67,
        1.0,
        1.0,
        (0, 0, 0),
    )

    # Tiny change in top_left
    after = paint_region(
        after,
        0.0,
        0.0,
        0.05,
        0.05,
        (200, 200, 200),
    )

    result = ScreenDiffAnalyzer().analyze(before, after)

    if len(result.changed_regions) >= 2:
        assert result.changed_regions[0] == "bottom_right"
