"""Integration tests — screen capture (runs on real OS)."""
import pytest
import numpy as np


@pytest.mark.integration
def test_capture_returns_rgb_array():
    """Must run on a real machine with a display."""
    from visionnav.perception.capture import ScreenCapture
    cap  = ScreenCapture()
    arr, meta = cap.capture()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 3
    assert arr.shape[2] == 3          # RGB channels
    assert meta["width"]  > 100
    assert meta["height"] > 100

@pytest.mark.integration
def test_capture_size_matches_metadata():
    from visionnav.perception.capture import ScreenCapture
    cap  = ScreenCapture()
    arr, meta = cap.capture()
    assert arr.shape[1] == meta["width"]
    assert arr.shape[0] == meta["height"]

@pytest.mark.integration
def test_screen_size_consistent():
    from visionnav.perception.capture import ScreenCapture
    cap = ScreenCapture()
    w, h = cap.get_screen_size()
    assert w > 0 and h > 0
