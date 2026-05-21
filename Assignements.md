# VisionNav Engineering Mentorship
### Session 1 of N — The Foundation

Welcome. From today you are not a student copying code. You are a junior engineer on a real AI startup team. I am your senior. We work together, I explain everything, you build everything.

Here is how every session works:

```
1. I teach one concept deeply
2. I show how it connects to VisionNav
3. You get a real coding assignment
4. You submit your code
5. I review it like a senior engineer
6. We move to the next topic
```

No shortcuts. No skipping. Every concept builds on the last.

---

## Before We Start — A Mindset Shift

Most people learn programming like this:
```
Copy code → run it → it works → move on
```

That produces someone who can follow tutorials but cannot build anything alone.

We learn like this:
```
Understand the problem → design a solution → write code → test it
→ break it intentionally → fix it → explain it to someone else
```

That produces an engineer.

**The test I use for every concept:**
> Can you explain this to a 12-year-old using only everyday objects as analogies?

If yes — you understand it. If no — you have memorized it, not learned it.

---

---

# SESSION 1 — What Is VisionNav? The Real Mental Model

Before we look at any code, you need the right mental model.
Most people think of VisionNav as "a program that clicks things."
That is wrong. That is like saying a car is "a machine that moves wheels."

**The correct mental model:**

```
VisionNav is a closed-loop perception-action system
with a learned policy for GUI environments.
```

Let me break that down into plain English.

---

## Concept 1 — What Is a Closed-Loop System?

**Open loop (no feedback):**
```
You tell a robot: "Walk 10 steps forward"
Robot walks 10 steps — even if there is a wall
Robot does not check if it hit the wall
```

**Closed loop (with feedback):**
```
You tell a robot: "Walk forward until you reach the door"
Robot takes one step → looks ahead → is that the door? No → takes another step
→ looks again → is that the door? Yes → stops
```

VisionNav is closed loop:
```
Agent takes action → captures new screenshot → checks if task progressed
→ decides next action based on what it sees → repeat
```

This is why we have the verifier. Every action is followed by a screenshot
to close the feedback loop. Without this, the agent is blind.

---

## Concept 2 — What Is a Perception-Action System?

Every intelligent system in the world has two parts:

```
PERCEPTION          ACTION
────────────────    ──────────────────
Eyes → brain        Brain → hands
Camera → computer   Computer → motors
Screenshot → VLM    VLM → mouse/keyboard
```

In VisionNav:
```
PERCEPTION SIDE:
  mss (screenshot) → OCR (text) → fusion → Observation object

ACTION SIDE:
  VLM output → parser → Action object → executor → OS
```

The Observation object is the bridge between perception and action.
It is literally what the agent "sees" before deciding what to do.

---

## Concept 3 — What Is a Learned Policy?

A **policy** is a function that maps observations to actions.

```
policy(observation) → action
```

Old automation (Selenium, RPA):
```
policy = if "Submit" button visible → click coordinates (400, 300)
         else if "Login" button visible → click coordinates (200, 500)
         ...
```
This is a **hardcoded policy** — brittle, breaks when UI changes.

VisionNav:
```
policy = Qwen2.5-VL-3B (fine-tuned neural network)
         policy(screenshot + task + history) → action
```
This is a **learned policy** — generalizes, handles new UIs it has never seen.

The entire training pipeline exists for one purpose: to learn a better policy.

---

## Concept 4 — The Six Modules and Their Roles

```
┌──────────────────────────────────────────────────────────────┐
│                    VisionNav System                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │PERCEPTION│→  │  POLICY  │→  │  ACTION  │                 │
│  │          │   │          │   │          │                 │
│  │mss       │   │Qwen VLM  │   │pyautogui │                 │
│  │Tesseract │   │(brain)   │   │ADB       │                 │
│  │fusion    │   │          │   │win32     │                 │
│  └──────────┘   └──────────┘   └──────────┘                 │
│       ↑                              ↓                       │
│  ┌──────────┐                  ┌──────────┐                 │
│  │VERIFIER  │←─────────────────│EXECUTOR  │                 │
│  │(did it   │                  │(does it) │                 │
│  │ change?) │                  │          │                 │
│  └──────────┘                  └──────────┘                 │
│                                                              │
│  ┌──────────┐   ┌──────────┐                                │
│  │ MEMORY   │   │ SAFETY   │                                │
│  │(SQLite)  │   │(guard)   │                                │
│  └──────────┘   └──────────┘                                │
└──────────────────────────────────────────────────────────────┘
```

**Each module has exactly one job:**
```
PERCEPTION  → Convert screen to structured data
POLICY      → Decide what action to take
ACTION      → Define what actions are possible (schema)
EXECUTOR    → Physically perform the action on the OS
VERIFIER    → Check if the action had the expected effect
MEMORY      → Remember what happened (for context + history)
SAFETY      → Block actions that are too dangerous
```

This is called **separation of concerns** — one of the most important principles
in software engineering. Each module is independently testable, replaceable,
and understandable.

---

## Concept 5 — Why the Architecture Is Built Around Interfaces

Open `src/visionnav/models/base.py` right now. Read it.

```python
class ModelBackend(Protocol):
    async def predict_action(
        self, observation, task, history, plan
    ) -> str: ...
```

This is a Protocol — a contract. It says:
> "Whatever you give me as a model, it must have this one method."

Why does this matter? Look at what we can now do:

```python
# Development — runs on your laptop CPU
model = LocalModelBackend(settings.model)

# Production — runs on GPU cluster
model = VLLMBackend(settings.model.vllm)

# Future — calls GPT-4o API
model = OpenAIBackend(settings.model.openai)

# Testing — returns fake output
model = MockModelBackend()
```

**All four of these work with the same agent loop.**
`agent.py` never changes. Only the model backend changes.

This is called the **Open/Closed Principle**:
> Open for extension (add new backends), closed for modification (don't change agent.py).

Without this principle, every time you change your model you would have to
rewrite the agent loop. That is what bad architecture looks like.

---

## Concept 6 — The Full Execution Flow (End to End)

Let me trace exactly what happens when you submit "Open Notepad":

```
POST /v1/tasks/ {"instruction": "Open Notepad"}
         ↓
FastAPI receives request → returns 202 immediately
         ↓
BackgroundTask spawns → run_task() starts in background
         ↓
VisionNavAgent.run("task-id", "Open Notepad") called
         ↓
planner.decompose("Open Notepad")
→ ["Find target app", "Click to open", "Wait for load", "Verify"]
         ↓
LOOP STARTS (step 0):
  1. platform.capture() → numpy array (1920×1080×3)
  2. ocr.run(array) → [TextRegion("EXPLORER",[0.08,0.1,0.13,0.11]), ...]
  3. platform.get_ui_tree() → [] (empty on Windows without setup)
  4. fuse(array, meta, regions, []) → Observation object
  5. model.predict_action(observation, "Open Notepad", [], plan)
     → "<think>I see desktop...</think><action>{type:key,key:win+r}</action>"
  6. parse_action(output) → Action(type=KEY, key="win+r")
  7. safety.classify(action) → RiskLevel.LOW → allowed
  8. before = current screenshot
  9. executor.execute(action, 1920, 1080) → pyautogui.hotkey("win","r")
  10. after = new screenshot
  11. verifier.verify(before, after, action) → (True, 0.097)
  12. memory.save_step(task_id, AgentState(...))
  13. action.type != DONE → continue to step 1
         ↓
LOOP continues until DONE or FAIL or max_steps
         ↓
memory.mark_task_complete(task_id, success, summary)
         ↓
GET /v1/tasks/task-id → {"status": "completed", "steps": 4}
```

That is the complete picture. Every file we built serves one specific step in this flow.

---

---

# MODULE DEEP DIVE 1 — Settings System

## Why Settings Deserve Serious Attention

Junior engineers treat settings as boring. Senior engineers know settings are critical.

A hardcoded value is a **time bomb**:
```python
# This worked in development
model = load_model("Qwen/Qwen2.5-VL-3B-Instruct")

# But when you deploy to production server:
# - Model path doesn't exist
# - You need a different quantization
# - Your API key is different
# - Your GPU count changed
# Result: Production crash, embarrassed, users angry
```

A settings system defuses the time bomb:
```python
# Development .env:
VISIONNAV_MODEL__NAME=Qwen/Qwen2.5-VL-3B-Instruct

# Production environment:
VISIONNAV_MODEL__NAME=/models/visionnav-3b-finetuned

# Same code, different behavior, zero changes to source code
model = load_model(settings.model.name)
```

## How Pydantic BaseSettings Works Internally

```python
class ModelSettings(BaseSettings):
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
```

When Python creates a `ModelSettings()` object it does this:

```
Step 1: Check environment variables
        VISIONNAV_MODEL__NAME exists? → use it
        VISIONNAV_MODEL__DTYPE exists? → use it

Step 2: Check .env file
        VISIONNAV_MODEL__NAME in .env? → use it (if not in environment)

Step 3: Use default values
        name = "Qwen/Qwen2.5-VL-3B-Instruct" (hardcoded default)
        dtype = "bfloat16"

Step 4: Validate types
        name must be str → ok
        dtype must be str → ok
        If wrong type → raise ValidationError immediately
```

The `__` double underscore in `VISIONNAV_MODEL__NAME` means "nested":
```
VISIONNAV_             → prefix (defined by env_prefix)
       MODEL__         → means settings.model
             NAME      → means .name attribute
```

So `VISIONNAV_MODEL__VLLM__BASE_URL` means `settings.model.vllm.base_url`.

---

---

# ASSIGNMENT 1 — The Settings Surgeon

## Your Task

You will create a **new configuration section** for VisionNav's OCR system
and write tests to prove it works correctly.

This assignment trains:
- Understanding of Pydantic BaseSettings deeply
- Environment variable handling
- Nested configuration design
- Test-driven development
- Configuration validation

---

## Requirements

### Part A — Extend Settings

Open `src/visionnav/settings.py`.

Add a new `OCRSettings` class and integrate it into `Settings`:

```python
class OCRSettings(BaseSettings):
    # Primary engine: "paddle" | "tesseract" | "auto"
    # "auto" = try paddle first, fall back to tesseract
    engine: str = "auto"

    # Minimum confidence to keep a text region (0.0 to 1.0)
    min_confidence: float = 0.6

    # Tesseract executable path (important for Windows)
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Maximum number of OCR regions to return
    max_regions: int = 50

    # Filter out regions shorter than this (removes single chars)
    min_text_length: int = 2

    # Language codes for OCR (comma-separated)
    languages: str = "en"
```

Rules for your implementation:
1. Add field-level validation: `engine` must be one of `"paddle"`, `"tesseract"`, `"auto"`
2. Add field-level validation: `min_confidence` must be between 0.0 and 1.0
3. Add field-level validation: `max_regions` must be between 1 and 200
4. If any validation fails, raise a `ValueError` with a clear message

---

### Part B — Wire Into OCR Engine

Open `src/visionnav/perception/ocr.py`.

The current `OCREngine.__init__` takes a hardcoded `min_confidence=0.5`.

Change it to accept an `OCRSettings` object:

```python
class OCREngine:
    def __init__(self, settings: OCRSettings | None = None) -> None:
        # If no settings provided, use defaults
        self._settings = settings or OCRSettings()
        # All config comes from settings, nothing hardcoded
```

Then update the `run()` method to use `self._settings` for:
- Which engine to try first (`engine` field)
- Confidence threshold (`min_confidence` field)
- Max regions limit (`max_regions` field)
- Minimum text length (`min_text_length` field)

---

### Part C — Write Tests

Create `tests/unit/test_ocr_settings.py` with these test cases:

```python
# Test 1: Default settings work correctly
def test_ocr_settings_defaults():
    s = OCRSettings()
    assert s.engine == "auto"
    assert s.min_confidence == 0.6
    assert s.max_regions == 50

# Test 2: Valid engine values accepted
def test_valid_engine_values():
    # These should NOT raise
    # test all three valid engines

# Test 3: Invalid engine raises ValueError
def test_invalid_engine_raises():
    # "invalid_engine" should raise ValueError

# Test 4: Confidence out of range raises
def test_confidence_below_zero_raises():
    # min_confidence=-0.1 should raise

def test_confidence_above_one_raises():
    # min_confidence=1.5 should raise

# Test 5: Max regions out of range
def test_max_regions_zero_raises():
    # max_regions=0 should raise

def test_max_regions_too_large_raises():
    # max_regions=201 should raise

# Test 6: Environment variable override works
def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("VISIONNAV_OCR__ENGINE", "tesseract")
    monkeypatch.setenv("VISIONNAV_OCR__MIN_CONFIDENCE", "0.8")
    s = OCRSettings()
    assert s.engine == "tesseract"
    assert s.min_confidence == 0.8

# Test 7: OCREngine respects settings
def test_ocr_engine_uses_settings():
    # Create OCREngine with custom settings
    # Create a test image with text
    # Run OCR
    # Verify regions returned respect max_regions limit
```

---

### Part D — Create a Test Image Generator

Write a utility function in `tests/unit/test_ocr_settings.py`:

```python
def create_test_image_with_text(texts: list[str]) -> np.ndarray:
    """
    Create a white image with the given texts placed at different positions.
    Use PIL to draw text.
    Returns numpy array (H, W, 3).
    """
    # Your implementation here
    # Hint: PIL.Image, PIL.ImageDraw, PIL.ImageFont
    # Place each text at a different y position
    # Return as numpy array
```

Use this function in your Test 7 above to create a real test image
with known text content, run OCR on it, and verify results.

---

## Success Criteria

Run all tests:
```bash
python -m pytest tests/unit/test_ocr_settings.py -v
```

All tests must pass. Then run the full test suite:
```bash
python -m pytest tests/unit/ -v
```

All 31+ existing tests must still pass. Your changes must not break anything.

---

## What I Will Review

When you submit, I will check:

```
Architecture quality:
  □ Does OCRSettings follow the same pattern as other Settings classes?
  □ Are validators written at the field level (using Pydantic validators)?
  □ Is OCREngine properly decoupled from hardcoded values?

Code quality:
  □ Are type hints on every function?
  □ Are variable names clear and descriptive?
  □ Is there a docstring on every class and non-trivial function?
  □ Are there no magic numbers (only settings values)?

Test quality:
  □ Does each test test exactly ONE thing?
  □ Are test names descriptive (what + scenario)?
  □ Does test_ocr_engine_uses_settings test BEHAVIOR not implementation?
  □ Is the test image generator a proper utility (reusable)?

Engineering mindset:
  □ Did you consider edge cases (empty text, whitespace-only text)?
  □ Did you think about what happens when tesseract is not installed?
```

---

## Hints (Read Only If Stuck for 30 Minutes)

**Hint for Pydantic validators:**
```python
from pydantic import field_validator

class OCRSettings(BaseSettings):
    engine: str = "auto"

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        allowed = {"paddle", "tesseract", "auto"}
        if v not in allowed:
            raise ValueError(f"engine must be one of {allowed}, got '{v}'")
        return v
```

**Hint for creating PIL image:**
```python
from PIL import Image, ImageDraw
img  = Image.new("RGB", (400, 200), color="white")
draw = ImageDraw.Draw(img)
draw.text((10, 50), "Hello World", fill="black")
arr  = np.array(img)
```

**Hint for environment variable test:**
```python
def test_env_override(monkeypatch):
    monkeypatch.setenv("VISIONNAV_OCR__ENGINE", "tesseract")
    # Note: create a NEW Settings() inside the test
    # because cached settings won't see the new env var
    s = OCRSettings()
    assert s.engine == "tesseract"
```

---

## Timeline

Do not rush. Do not ask me for help until you have tried for at least 30 minutes.

When you are stuck, tell me:
```
File: [which file]
Line: [which line approximately]
Problem: [what exactly is not working]
Tried: [what you already attempted]
```

When you are done, paste your code here and say **"Assignment 1 submitted."**

Then I will review it and we move to the next topic.

---

## What Comes After This Assignment

```
After Assignment 1:  Deep dive into async/await (why it matters for agents)
After Assignment 2:  Deep dive into the perception pipeline
After Assignment 3:  Deep dive into the action system
After Assignment 4:  Deep dive into the agent loop internals
After Assignment 5:  Begin Phase 16 — Dataset Factory
  → How world-class datasets are designed
  → How to build an annotation pipeline from scratch
  → How Urdu/Pashto becomes our competitive moat
  → Coding a real data collection tool for VisionNav
```

---

One last thing before you start.

Read these two lines and understand them before writing any code:

> **"A test is not proof that code works. A test is a specification of what code should do."**

> **"If you cannot explain every line of your code without looking at it, you don't own that code yet."**

Now start Assignment 1. Come back when it is done.