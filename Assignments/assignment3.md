# SESSION 3 — The Perception Pipeline: How Agents See

---

## The Central Question

When you look at a screen, your brain instantly knows:
- Where every button is
- What the text says
- Which elements are clickable
- What changed since one second ago

How do we give a machine the same capability?
That is the entire perception pipeline.

---

## Concept 1 — What a Screenshot Actually Is

You think a screenshot is an image. Wrong.
A screenshot is a **3-dimensional array of integers**.

```python
import mss
import numpy as np

with mss.MSS() as sct:
    shot = sct.grab(sct.monitors[1])
    arr  = np.frombuffer(shot.rgb, dtype=np.uint8)
    arr  = arr.reshape(shot.height, shot.width, 3)

print(arr.shape)    # (1080, 1920, 3)
print(arr.dtype)    # uint8 (0 to 255)
```

The shape means:
```
(1080, 1920, 3)
  ↑      ↑    ↑
  rows  cols  channels (Red, Green, Blue)
```

Every pixel is three numbers:
```
arr[0][0]       = [255, 255, 255]  ← top-left pixel, white
arr[540][960]   = [30,  144, 255]  ← center pixel, blue
arr[1079][1919] = [0,   0,   0]    ← bottom-right pixel, black
```

**Why does this matter for the agent?**

The verifier compares screenshots using numpy:
```python
diff         = np.abs(before.astype(np.int16) - after.astype(np.int16))
change_ratio = float((diff > 30).mean())
```

Line by line:
```
before.astype(np.int16)          → convert uint8 to int16 to allow negative subtraction
after.astype(np.int16)           → same
np.abs(before - after)           → absolute difference per pixel per channel
diff > 30                        → True where pixel changed by more than 30 intensity units
.mean()                          → fraction of pixels that changed
```

This is computer vision without a model. Pure math on arrays.

---

## Concept 2 — How OCR Works Internally

OCR stands for Optical Character Recognition.
Simple explanation:

```
Step 1 — Text Detection:
  Where are the text regions on this image?
  PaddleOCR runs a neural network that outputs bounding boxes.
  Output: [(x1,y1,x2,y2), (x1,y1,x2,y2), ...]

Step 2 — Text Recognition:
  For each detected region — what letters are in it?
  Another neural network reads the cropped region.
  Output: "Submit" with confidence 0.97

Step 3 — Combined:
  TextRegion(text="Submit", bbox=[0.7, 0.88, 0.85, 0.93], confidence=0.97)
```

PaddleOCR runs both steps internally. You call `.ocr(image)` and get both.

**The coordinates problem:**

Raw PaddleOCR returns pixel coordinates:
```
[[45, 120], [230, 120], [230, 145], [45, 145]]
```
That is a quadrilateral (4 corner points) in pixels.

We convert to our normalized axis-aligned bbox format:
```python
xs = [p[0] for p in pts]   # [45, 230, 230, 45]
ys = [p[1] for p in pts]   # [120, 120, 145, 145]

# Normalize by image dimensions
x1 = min(xs) / image_width    # 45  / 1920 = 0.023
y1 = min(ys) / image_height   # 120 / 1080 = 0.111
x2 = max(xs) / image_width    # 230 / 1920 = 0.119
y2 = max(ys) / image_height   # 145 / 1080 = 0.134
```

Now the bbox `[0.023, 0.111, 0.119, 0.134]` works on any screen size.

---

## Concept 3 — The Observation Object as a Contract

The `Observation` class is the most important data structure in perception.
It is the contract between perception and reasoning.

```python
@dataclass
class Observation:
    screenshot_b64: str           # what the model "sees" (image)
    screenshot_path: str          # where it is stored on disk
    screen_width: int             # needed to convert coords back to pixels
    screen_height: int            # needed to convert coords back to pixels
    ocr_regions: list[TextRegion] # what text is on screen (structured)
    ui_elements: list[dict]       # what UI elements exist (accessibility)
    platform: str                 # windows / macos / linux / android
```

Why does it contain both `screenshot_b64` AND `ocr_regions`?

Because the VLM needs both:
```
screenshot_b64  → the visual input (model sees image directly)
ocr_regions     → pre-computed text (reduces model's OCR burden)
                  model can focus on reasoning, not re-reading text
```

The `to_text_summary()` method converts `ocr_regions` to a text prompt:
```
Detected text on screen:
  - 'Submit' at [0.70,0.88,0.85,0.93]
  - 'Cancel' at [0.50,0.88,0.65,0.93]
  - 'Email address' at [0.10,0.40,0.90,0.47]
```

This text is injected into the VLM prompt alongside the image.
The model sees both — the image visually AND the structured text list.
This doubles the accuracy of element grounding.

---

## Concept 4 — The Action System Architecture

After perception comes reasoning. After reasoning comes action.

The action system has four files, each with one job:

```
schema.py   → DEFINE:   what actions exist and what they contain
parser.py   → PARSE:    convert model text output to typed Action object
executor.py → DISPATCH: route Action to the right platform method
verifier.py → VERIFY:   did the action have the expected effect?
```

### schema.py — The Action Vocabulary

```python
class ActionType(str, Enum):
    CLICK        = "click"
    DOUBLE_CLICK = "double_click"
    TYPE         = "type"
    KEY          = "key"
    SCROLL       = "scroll"
    WAIT         = "wait"
    DONE         = "done"
    FAIL         = "fail"

class Action(BaseModel):
    type:        ActionType
    coordinates: tuple[float, float] | None  # normalized [0,1]
    text:        str | None
    key:         str | None
    description: str = ""
    confidence:  float = 1.0
```

Using `str, Enum` means the enum serializes to its string value:
```python
ActionType.CLICK == "click"   # True
json.dumps({"type": ActionType.CLICK})   # '{"type": "click"}'
```

This matters for database storage and API responses — no conversion needed.

### parser.py — The Most Defensive Code in the Project

The parser is the bridge between the model's text output and typed Python objects.
The model outputs raw text — the parser must handle everything that can go wrong.

```python
def parse_action(model_output: str) -> Action:
    # Step 1: Find the <action> block
    match = _PATTERN.search(model_output)
    if not match:
        raise ActionParseError("No <action> block found")

    # Step 2: Parse JSON inside the block
    try:
        raw = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        raise ActionParseError(f"Invalid JSON: {exc}")

    # Step 3: Validate action type is known
    try:
        action_type = ActionType(raw.get("type", ""))
    except ValueError:
        raise ActionParseError(f"Unknown type: {raw.get('type')}")

    # Step 4: Validate coordinates if present
    coords = raw.get("coordinates")
    if coords is not None:
        if not (isinstance(coords, (list, tuple)) and len(coords) == 2):
            raise ActionParseError("coordinates must be [x, y]")
        nx, ny = float(coords[0]), float(coords[1])
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise ActionParseError(f"coordinates out of [0,1]: [{nx},{ny}]")
        coords = (nx, ny)

    return Action(type=action_type, coordinates=coords, ...)
```

Notice every failure raises `ActionParseError` — a specific exception.
The agent catches `ActionParseError` and creates a FAIL action.
It never crashes. It always continues.

### executor.py — The Dispatcher

```python
async def _dispatch(self, action: Action, w: int, h: int) -> bool:
    match action.type:
        case ActionType.CLICK:
            x, y = self._abs(action, w, h)
            return await self._platform.execute_click(x, y, "left")
        case ActionType.TYPE:
            return await self._platform.execute_type(action.text or "")
        case ActionType.KEY:
            return await self._platform.execute_key(action.key or "")
        ...
```

This is a **dispatcher pattern** — one function routes to many handlers
based on the action type. Clean, readable, easily extended.

To add a new action type:
1. Add it to `ActionType` enum in `schema.py`
2. Add a case to `_dispatch` in `executor.py`
3. Add a method to `PlatformAdapter` in `base.py`
4. That is it. Zero other changes.

### verifier.py — Pixel-Level Change Detection

```python
def verify(self, before, after, action) -> tuple[bool, float]:
    if action.type in _NO_VERIFY:
        return True, 0.0

    diff         = np.abs(before.astype(np.int16) - after.astype(np.int16))
    change_ratio = float((diff > 30).mean())
    return change_ratio >= self._threshold, change_ratio
```

The `> 30` threshold filters out noise — minor anti-aliasing, clock updates,
cursor blink. Real UI changes cause much larger pixel differences.

---

---

# ASSIGNMENT 3 — The Screen Intelligence System

## What You Are Building

A **ScreenDiff analyzer** — a production-grade module that does not just
detect IF the screen changed, but classifies HOW it changed.

This is the foundation of the agent's verification intelligence.
Instead of "change ratio = 0.4" the agent will know
"a new window appeared in the top-right quarter of the screen."

This assignment trains:
- Numpy array operations (real computer vision)
- Class design (clean object-oriented code)
- Enum-based classification
- Region-of-interest analysis
- Structured data design
- Testing visual logic

---

## What You Will Build

A new file: `src/visionnav/actions/screen_diff.py`

---

## The ScreenDiff System Design

### The Data Structures

```python
from enum import Enum
from dataclasses import dataclass
import numpy as np


class ChangeType(Enum):
    """Classification of how a screen changed."""
    NO_CHANGE       = "no_change"       # nothing changed
    MINOR_CHANGE    = "minor_change"    # < 2% pixels changed (cursor, clock)
    TEXT_UPDATE     = "text_update"     # text region changed
    NEW_ELEMENT     = "new_element"     # new UI element appeared
    ELEMENT_REMOVED = "element_removed" # UI element disappeared
    LAYOUT_SHIFT    = "layout_shift"    # major layout change (new window/dialog)
    FULL_SCREEN     = "full_screen"     # > 50% screen changed (app switch)


@dataclass(frozen=True)
class ScreenRegion:
    """A rectangular region of the screen in normalized coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    name: str   # human-readable: "top_left", "center", "bottom_right" etc.

    def to_pixel_slice(self, width: int, height: int):
        """Convert normalized region to numpy array slice."""
        r1 = int(self.y1 * height)
        r2 = int(self.y2 * height)
        c1 = int(self.x1 * width)
        c2 = int(self.x2 * width)
        return slice(r1, r2), slice(c1, c2)


@dataclass(frozen=True)
class DiffResult:
    """Complete result of comparing two screenshots."""
    change_type:       ChangeType
    change_ratio:      float          # 0.0 to 1.0 — fraction of pixels changed
    changed_regions:   list[str]      # names of ScreenRegions with most change
    most_changed_area: ScreenRegion   # the single region with most change
    is_significant:    bool           # True if change_ratio > threshold
    summary:           str            # human-readable: "New dialog appeared top-center"
```

### The Analyzer Class

```python
class ScreenDiffAnalyzer:
    """
    Analyzes differences between two screenshots.

    Beyond simple pixel comparison:
    - WHERE on screen did change occur?
    - WHAT TYPE of change is it?
    - IS the change significant enough to indicate action succeeded?

    This gives the agent much richer feedback than a single number.
    """

    # Nine regions that divide the screen (3x3 grid)
    REGIONS: list[ScreenRegion] = [
        ScreenRegion(0.0,  0.0,  0.33, 0.33, "top_left"),
        ScreenRegion(0.33, 0.0,  0.67, 0.33, "top_center"),
        ScreenRegion(0.67, 0.0,  1.0,  0.33, "top_right"),
        ScreenRegion(0.0,  0.33, 0.33, 0.67, "middle_left"),
        ScreenRegion(0.33, 0.33, 0.67, 0.67, "center"),
        ScreenRegion(0.67, 0.33, 1.0,  0.67, "middle_right"),
        ScreenRegion(0.0,  0.67, 0.33, 1.0,  "bottom_left"),
        ScreenRegion(0.33, 0.67, 0.67, 1.0,  "bottom_center"),
        ScreenRegion(0.67, 0.67, 1.0,  1.0,  "bottom_right"),
    ]

    def __init__(self, threshold: float = 0.01) -> None:
        self._threshold = threshold

    def analyze(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> DiffResult:
        """
        Full analysis of screen change between two screenshots.

        Args:
            before: Screenshot before the action (H, W, 3) uint8
            after:  Screenshot after the action  (H, W, 3) uint8

        Returns:
            DiffResult with complete change analysis
        """
        # Your implementation here

    def _compute_diff_map(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-pixel change intensity.

        Returns float array (H, W) where each value is
        the average absolute change across RGB channels (0.0 to 255.0).
        """
        # Your implementation here

    def _analyze_regions(
        self,
        diff_map: np.ndarray,
    ) -> tuple[list[str], ScreenRegion]:
        """
        Find which screen regions changed the most.

        Returns:
            changed_regions:   list of region names sorted by change intensity
            most_changed_area: the single region with highest change ratio
        """
        # Your implementation here

    def _classify_change(
        self,
        change_ratio: float,
        region_ratios: dict[str, float],
    ) -> ChangeType:
        """
        Classify the TYPE of change based on:
        - Overall change ratio
        - Pattern of which regions changed

        Classification rules (implement all of these):

        NO_CHANGE:       change_ratio < 0.001
        MINOR_CHANGE:    change_ratio < 0.02
        FULL_SCREEN:     change_ratio > 0.50
        LAYOUT_SHIFT:    change_ratio > 0.15 AND change spread across 5+ regions
        NEW_ELEMENT:     change concentrated in 1-2 regions AND ratio 0.05-0.15
        ELEMENT_REMOVED: same as NEW_ELEMENT (visually same pattern)
        TEXT_UPDATE:     change_ratio 0.02-0.05 (small but real)
        """
        # Your implementation here

    def _generate_summary(
        self,
        change_type: ChangeType,
        changed_regions: list[str],
        change_ratio: float,
    ) -> str:
        """
        Generate a human-readable summary of what changed.

        Examples:
          "No meaningful change detected"
          "Minor change in top_right (likely cursor or clock)"
          "New element appeared in center"
          "Major layout shift — possible new window (top_left, top_center, middle_left)"
          "Full screen change — application switched"
        """
        # Your implementation here
```

---

## Implementation Requirements

### `_compute_diff_map`:
```
Input:  before (H, W, 3), after (H, W, 3)
Output: diff (H, W) float

Computation:
  1. Convert both to float32 (prevent overflow)
  2. Take absolute difference
  3. Average across the 3 color channels (axis=2)
  4. Result: one float per pixel representing total change intensity
```

### `_analyze_regions`:
```
For each of the 9 REGIONS:
  1. Extract that region from the diff_map using to_pixel_slice()
  2. Compute: what fraction of pixels in this region changed by > 30?
  3. Record: region name → change ratio

Sort regions by change ratio (highest first)
Return: list of names (sorted), most changed ScreenRegion object
```

### `_classify_change`:
```
Implement the 7 classification rules in order:
  Most specific first (FULL_SCREEN is obvious)
  Most subtle last (TEXT_UPDATE is hardest to detect)

Count "active regions" = regions with change_ratio > 0.05

Rules (in this exact priority order):
  1. change_ratio < 0.001 → NO_CHANGE
  2. change_ratio < 0.02  → MINOR_CHANGE
  3. change_ratio > 0.50  → FULL_SCREEN
  4. change_ratio > 0.15 AND active_regions >= 5 → LAYOUT_SHIFT
  5. 0.05 <= change_ratio <= 0.15 AND active_regions <= 2 → NEW_ELEMENT
  6. 0.02 <= change_ratio <= 0.05 → TEXT_UPDATE
  7. Everything else → MINOR_CHANGE (safe default)
```

### `analyze` (the main method):
```
1. Validate: both arrays must be same shape. If not → return FULL_SCREEN result
2. Call _compute_diff_map
3. Compute overall change_ratio from diff_map
4. Call _analyze_regions to get region breakdown
5. Build region_ratios dict {region_name: ratio}
6. Call _classify_change
7. Call _generate_summary
8. Build and return DiffResult
```

---

## Tests to Write

Create `tests/unit/test_screen_diff.py`:

```python
# Use this helper to create test images
def make_screen(height=100, width=200, color=(255,255,255)) -> np.ndarray:
    return np.full((height, width, 3), color, dtype=np.uint8)

def paint_region(screen, y1_ratio, x1_ratio, y2_ratio, x2_ratio, color):
    """Paint a rectangular area on a screen array."""
    h, w = screen.shape[:2]
    r1, r2 = int(y1_ratio*h), int(y2_ratio*h)
    c1, c2 = int(x1_ratio*w), int(x2_ratio*w)
    screen[r1:r2, c1:c2] = color
    return screen


# Test 1: Identical screens → NO_CHANGE
def test_identical_screens():
    screen = make_screen()
    analyzer = ScreenDiffAnalyzer()
    result = analyzer.analyze(screen, screen.copy())
    assert result.change_type == ChangeType.NO_CHANGE
    assert result.change_ratio == 0.0
    assert result.is_significant is False


# Test 2: Completely different screens → FULL_SCREEN
def test_full_screen_change():
    before = make_screen(color=(255,255,255))  # white
    after  = make_screen(color=(0,0,0))        # black
    result = ScreenDiffAnalyzer().analyze(before, after)
    assert result.change_type == ChangeType.FULL_SCREEN
    assert result.change_ratio > 0.9


# Test 3: Small change in one region → NEW_ELEMENT
def test_new_element_detected():
    before = make_screen()
    after  = make_screen()
    # Paint a small black rectangle in center
    after  = paint_region(after, 0.4, 0.4, 0.6, 0.6, (0,0,0))
    result = ScreenDiffAnalyzer().analyze(before, after)
    assert result.change_type in (ChangeType.NEW_ELEMENT, ChangeType.TEXT_UPDATE)
    assert "center" in result.changed_regions


# Test 4: Very tiny change → MINOR_CHANGE
def test_minor_change():
    before = make_screen()
    after  = make_screen()
    # Change only 3 pixels (simulates cursor blink)
    after[5][5] = [200, 200, 200]
    after[5][6] = [200, 200, 200]
    after[5][7] = [200, 200, 200]
    result = ScreenDiffAnalyzer().analyze(before, after)
    assert result.change_type in (ChangeType.NO_CHANGE, ChangeType.MINOR_CHANGE)


# Test 5: most_changed_area is correct region
def test_most_changed_area_region():
    before = make_screen()
    after  = make_screen()
    # Paint bottom_right region
    after  = paint_region(after, 0.7, 0.7, 1.0, 1.0, (0,0,0))
    result = ScreenDiffAnalyzer().analyze(before, after)
    assert result.most_changed_area.name == "bottom_right"


# Test 6: DiffResult is immutable (frozen dataclass)
def test_diff_result_immutable():
    before = make_screen()
    after  = make_screen(color=(0,0,0))
    result = ScreenDiffAnalyzer().analyze(before, after)
    with pytest.raises((AttributeError, TypeError)):
        result.change_ratio = 0.0   # frozen dataclass should reject this


# Test 7: Different shape arrays are handled gracefully
def test_different_shape_handled():
    before = make_screen(height=100, width=200)
    after  = make_screen(height=200, width=400)
    result = ScreenDiffAnalyzer().analyze(before, after)
    # Should not crash — return FULL_SCREEN as safe default
    assert result.change_type == ChangeType.FULL_SCREEN


# Test 8: Summary is human-readable string
def test_summary_is_string():
    before = make_screen()
    after  = make_screen(color=(0,0,0))
    result = ScreenDiffAnalyzer().analyze(before, after)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 5


# Test 9: changed_regions is sorted by change intensity (most changed first)
def test_changed_regions_sorted():
    before = make_screen(height=300, width=300)
    after  = make_screen(height=300, width=300)
    # Paint bottom_right heavily, top_left lightly
    after = paint_region(after, 0.67, 0.67, 1.0, 1.0, (0,0,0))    # big change
    after = paint_region(after, 0.0,  0.0,  0.05, 0.05, (200,200,200))  # tiny change
    result = ScreenDiffAnalyzer().analyze(before, after)
    if len(result.changed_regions) >= 2:
        assert result.changed_regions[0] == "bottom_right"
```

---

## Success Criteria

```bash
python -m pytest tests/unit/test_screen_diff.py -v
```

All 9 tests pass. Then:

```bash
python -m pytest tests/unit/ -v
```

All existing tests still pass. Nothing broken.

---

## What I Will Review

```
Architecture:
  □ Is ScreenRegion correctly immutable (frozen=True)?
  □ Is DiffResult correctly immutable (frozen=True)?
  □ Does to_pixel_slice work correctly with numpy slicing?
  □ Is _classify_change logic in correct priority order?

Computer Vision Logic:
  □ Does _compute_diff_map average across channels correctly?
  □ Does _analyze_regions correctly extract sub-arrays with to_pixel_slice?
  □ Does the region sorting correctly rank by change intensity?

Code Quality:
  □ Type hints on every method parameter and return value?
  □ Docstring on every method?
  □ No magic numbers (use named constants or comments)?
  □ Clean readable logic (not one-liners that need 10 minutes to decode)?

Tests:
  □ Does each test use make_screen and paint_region properly?
  □ Does test_different_shape_handled test graceful degradation?
  □ Does test_changed_regions_sorted use a large enough image?
    (small images may not have enough pixels for reliable testing)
```

---

## One Concept To Remember

```
The diff_map reduces a 3D problem (H × W × 3) to a 2D problem (H × W).

3D: each pixel has R, G, B channels  → complex
2D: each pixel has one change score  → simple to analyze

This reduction is called "projection" in computer vision.
By averaging across channels, we project the color space
onto a single "intensity of change" axis.

Almost every image processing algorithm starts with this kind of
dimensionality reduction — making the data simpler before analyzing it.
```

---

Come back with both Assignment 2 and Assignment 3 when ready.

Say **"Assignments 2 and 3 submitted"** and paste both code sets.
We review both, then move to Assignment 4 — the agent loop internals.