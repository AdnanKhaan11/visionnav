"""Image helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def save_screenshot(image: np.ndarray, path: Path | str) -> None:
    from PIL import Image
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(p, "PNG")


def encode_base64(image: np.ndarray) -> str:
    import base64, io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()
