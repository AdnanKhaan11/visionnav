"""Coordinate normalisation / denormalisation."""
from __future__ import annotations


def normalize(x: int, y: int, w: int, h: int) -> tuple[float, float]:
    return round(x / w, 4), round(y / h, 4)

def denormalize(nx: float, ny: float, w: int, h: int) -> tuple[int, int]:
    return int(nx * w), int(ny * h)

def validate_normalized(nx: float, ny: float) -> bool:
    return 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0

def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2
