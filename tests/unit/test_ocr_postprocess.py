"""Unit tests — OCR output validation and filtering."""
from visionnav.perception.ocr import TextRegion


def test_text_region_fields():
    r = TextRegion(text="Submit", bbox=(0.1, 0.1, 0.4, 0.15), confidence=0.95)
    assert r.text == "Submit"
    assert r.confidence == 0.95

def test_bbox_normalised():
    r = TextRegion(text="OK", bbox=(0.0, 0.0, 1.0, 1.0), confidence=0.8)
    x1, y1, x2, y2 = r.bbox
    assert 0.0 <= x1 <= 1.0
    assert 0.0 <= y1 <= 1.0
    assert x2 >= x1
    assert y2 >= y1

def test_low_confidence_check():
    r = TextRegion(text="blurry", bbox=(0.0, 0.0, 0.1, 0.05), confidence=0.3)
    assert r.confidence < 0.5   # Would be filtered by OCREngine min_confidence
