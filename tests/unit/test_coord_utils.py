"""Unit tests — coordinate utilities."""
from visionnav.utils.coords import normalize, denormalize, validate_normalized, bbox_center


def test_normalize_center():
    assert normalize(640, 360, 1280, 720) == (0.5, 0.5)

def test_normalize_origin():
    assert normalize(0, 0, 1280, 720) == (0.0, 0.0)

def test_round_trip():
    x, y = 320, 240
    nx, ny = normalize(x, y, 1280, 720)
    assert denormalize(nx, ny, 1280, 720) == (x, y)

def test_validate_valid():
    assert validate_normalized(0.0, 0.0) is True
    assert validate_normalized(1.0, 1.0) is True

def test_validate_invalid():
    assert validate_normalized(1.1, 0.5) is False
    assert validate_normalized(-0.1, 0.5) is False

def test_bbox_center():
    cx, cy = bbox_center(0.1, 0.1, 0.5, 0.5)
    assert cx == 0.3 and cy == 0.3
