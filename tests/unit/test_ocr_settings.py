import pytest
import numpy as np

from PIL import Image, ImageDraw

from visionnav.settings import OCRSettings, Settings
from visionnav.perception.ocr import OCREngine


# Test 1: Default settings work correctly
def test_ocr_settings_defaults():

    s = OCRSettings()

    assert s.engine == "auto"
    assert s.min_confidence == 0.5
    assert s.max_regions == 50


# Test 2: Valid engine values accepted
def test_valid_engine_values():

    OCRSettings(engine="auto")
    OCRSettings(engine="paddle")
    OCRSettings(engine="tesseract")


# Test 3: Invalid engine raises ValueError
def test_invalid_engine_raises():

    with pytest.raises(ValueError):

        OCRSettings(engine="invalid_engine")


# Test 4: Confidence out of range raises
def test_confidence_below_zero_raises():

    with pytest.raises(ValueError):

        OCRSettings(min_confidence=-0.1)


def test_confidence_above_one_raises():

    with pytest.raises(ValueError):

        OCRSettings(min_confidence=1.5)


# Test 5: Max regions out of range
def test_max_regions_zero_raises():

    with pytest.raises(ValueError):

        OCRSettings(max_regions=0)


def test_max_regions_too_large_raises():

    with pytest.raises(ValueError):

        OCRSettings(max_regions=201)


# Test 6: Environment variable override works
def test_env_var_overrides_default(monkeypatch):

    monkeypatch.setenv(
        "VISIONNAV_OCR__ENGINE",
        "tesseract",
    )

    monkeypatch.setenv(
        "VISIONNAV_OCR__MIN_CONFIDENCE",
        "0.8",
    )

    # Use parent Settings object because
    # nested env vars belong to Settings
    s = Settings().ocr

    assert s.engine == "tesseract"
    assert s.min_confidence == 0.8


def create_test_image_with_text(
    texts: list[str],
) -> np.ndarray:
    """
    Create a white image with multiple text lines.
    Returns numpy array (H, W, 3).
    """

    image = Image.new(
        "RGB",
        (800, 400),
        color="white",
    )

    draw = ImageDraw.Draw(image)

    y = 20

    for text in texts:

        draw.text(
            (20, y),
            text,
            fill="black",
        )

        y += 50

    return np.array(image)


# Test 7: OCREngine respects settings
def test_ocr_engine_uses_settings():

    settings = OCRSettings(
        max_regions=2,
    )

    engine = OCREngine(settings)

    # Create image with MANY text regions
    image = create_test_image_with_text(
        [
            "Region One",
            "Region Two",
            "Region Three",
            "Region Four",
            "Region Five",
        ]
    )

    regions = engine.run(image)

    # OCR output must respect max_regions
    assert len(regions) <= 2
