import sys

sys.path.insert(0, "src")

from visionnav.perception.capture import ScreenCapture
from visionnav.perception.ocr import OCREngine
from visionnav.perception.ui_tree import get_ui_tree
from visionnav.perception.fusion import fuse

print("=" * 50)
print("VisionNav Perception Pipeline Test")
print("=" * 50)

# Step 1 - Capture screen
print("\nStep 1: Capturing screen...")
cap = ScreenCapture()
arr, meta = cap.capture()
print(f"  Screen captured: {meta['width']}x{meta['height']}")

# Step 2 - Run OCR
print("\nStep 2: Running OCR...")
ocr = OCREngine()
regions = ocr.run(arr)
print(f"  Text regions found: {len(regions)}")

# Step 3 - Get UI tree
print("\nStep 3: Getting UI tree...")
ui = get_ui_tree()
print(f"  UI elements found: {len(ui)}")

# Step 4 - Fuse everything
print("\nStep 4: Building observation...")
obs = fuse(arr, meta, regions, ui)
print(f"  Observation created successfully")
print(f"  Screenshot encoded: {len(obs.screenshot_b64)} characters")

# Step 5 - Show what agent sees
print("\nStep 5: What agent sees:")
print("-" * 40)
summary = obs.to_text_summary()
print(summary[:500])
print("-" * 40)

print("\nPerception pipeline working correctly!")
