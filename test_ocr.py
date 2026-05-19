import sys

sys.path.insert(0, "src")

from visionnav.perception.capture import ScreenCapture
from visionnav.perception.ocr import OCREngine

cap = ScreenCapture()
ocr = OCREngine()

print("Capturing your screen...")
arr, meta = cap.capture()
print(f"Screen size: {meta['width']}x{meta['height']}")

print("Running OCR...")
regions = ocr.run(arr)
print(f"Text regions found: {len(regions)}")
print()
print("Text on your screen right now:")

for r in regions:
    print(f"  {r.text}  (confidence: {r.confidence:.2f})")
