# VisionNav — Complete Development Plan
### Phase by Phase. Day by Day. Nothing Skipped.
**Start Date:** Tomorrow 8:00 AM
**Goal:** Fully working AI GUI agent with fine-tuned model

---

## Current Status (End of Day 1)

```
✅ Project structure created (77+ files)
✅ Virtual environment setup (uv)
✅ Dependencies installed
✅ 31 unit tests passing
✅ FastAPI server running
✅ Small model (DialoGPT) working
✅ Full agent loop completed first task
✅ Code pushed to GitHub
✅ Colab setup guide ready

❌ OCR not reading screen properly (Tesseract PATH)
❌ Model not trained on GUI data
❌ Agent makes wrong decisions (untrained)
❌ No real GUI automation tested
```

---

## The Big Picture

```
Phase 1  → Fix & Stabilize (Days 2-3)
Phase 2  → Perception Working (Days 4-5)
Phase 3  → Data Pipeline (Days 6-10)
Phase 4  → Model Fine-Tuning (Days 11-20)
Phase 5  → Automation Engine (Days 21-25)
Phase 6  → Full Agent Loop (Days 26-30)
Phase 7  → API Polish (Days 31-35)
Phase 8  → Deployment (Days 36-40)
Phase 9  → Apps & Future (Day 41+)
```

---

---

# PHASE 1 — Fix & Stabilize
### Days 2-3 | Goal: Everything works cleanly with small model

---

## Day 2 — Morning (8:00 AM - 12:00 PM)

### Task 1 — Fix OCR Permanently

**Problem:** Tesseract PATH not working in project terminal.

**What to do:**

Open `D:\visionnav\src\visionnav\perception\ocr.py`

Find `_run_tesseract` function. Make sure this line is there:

```python
pytesseract.pytesseract.tesseract_cmd = r"D:\mlops-tools\Tesseract-OCR\tesseract.exe"
```

**Test it:**
```bash
python -c "
import sys
sys.path.insert(0, 'src')
import numpy as np
from PIL import Image, ImageDraw
from visionnav.perception.ocr import OCREngine

# Create test image with text
img = Image.new('RGB', (400, 100), color='white')
draw = ImageDraw.Draw(img)
draw.text((10, 30), 'Hello VisionNav OCR Test', fill='black')
arr = np.array(img)

ocr = OCREngine()
regions = ocr.run(arr)
print(f'Found {len(regions)} text regions')
for r in regions:
    print(f'  Text: {r.text}  Confidence: {r.confidence:.2f}')
"
```

**Expected output:**
```
Found 1 text regions
  Text: Hello VisionNav OCR Test  Confidence: 0.95
```

**Checkpoint:** OCR reads text correctly before moving forward.

---

### Task 2 — Fix Action Null In Database

**Problem:** When we query task steps, action shows `null`.

**What to do:**

Open `D:\visionnav\src\visionnav\memory\sqlite.py`

Find `_to_state` function at the bottom.

Replace it with:

```python
def _to_state(data_json: str) -> AgentState:
    from datetime import datetime
    from visionnav.actions.schema import Action, ActionType
    d = json.loads(data_json)

    action = None
    if d.get("action_type"):
        try:
            action = Action(
                type=ActionType(d["action_type"]),
                description=d.get("action_description", ""),
            )
        except Exception:
            action = None

    return AgentState(
        step_index=d["step_index"],
        task_instruction=d["task_instruction"],
        screenshot_path=d["screenshot_path"],
        ocr_text=d["ocr_text"],
        action_taken=action,
        action_success=d["action_success"],
        reasoning=d["reasoning"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        error=d.get("error"),
    )
```

Also update `save_step` to save action description:

```python
data = {
    ...
    "action_type": state.action_taken.type if state.action_taken else None,
    "action_description": state.action_taken.description if state.action_taken else None,
    ...
}
```

**Test it:** Submit a task. Check result. Action should show `done` not `null`.

---

### Task 3 — Run All Tests Again

```bash
python -m pytest tests/unit/ -v
```

All 31 must pass. Fix any failures before moving forward.

---

## Day 2 — Afternoon (1:00 PM - 5:00 PM)

### Task 4 — Test Agent On 5 Real Tasks

Submit these 5 tasks one by one from `/docs`:

```
Task 1: "Take a screenshot of the current screen"
Task 2: "Click somewhere on the screen"
Task 3: "Open Notepad"
Task 4: "Type Hello World"
Task 5: "Scroll down the page"
```

For each task:
1. Submit it
2. Check terminal output
3. Query task result
4. Write down what the model responded

**Goal:** Understand what the small model outputs for different instructions.

---

### Task 5 — Understand Model Output Quality

After testing 5 tasks answer these questions:

```
1. Does the model always output <action> block correctly?
2. Does the action type make sense for the task?
3. Does the parser handle the output without errors?
4. What percentage of tasks complete successfully?
```

Write your answers in a file `notes/day2_observations.md`

---

## Day 3 — Full Day

### Task 6 — Write Integration Tests

Open `D:\visionnav\tests\integration\test_agent_loop.py`

Add these tests:

```python
@pytest.mark.integration
async def test_agent_saves_steps_to_db(mock_model, mock_platform, tmp_path, mocker):
    """Steps must be saved to database after each action."""
    from visionnav.memory.sqlite import SQLiteMemoryStore
    memory = SQLiteMemoryStore(f"sqlite+aiosqlite:///{tmp_path}/t.db")
    agent  = make_agent(mock_model, mock_platform, tmp_path, mocker)
    result = await agent.run("test-001", "Open Chrome")
    steps  = await memory.get_task_history("test-001")
    assert len(steps) >= 1

@pytest.mark.integration
async def test_agent_handles_invalid_model_output(mock_model, mock_platform, tmp_path, mocker):
    """Agent must not crash on bad model output."""
    mock_model.predict_action.return_value = "this is not valid output at all"
    agent  = make_agent(mock_model, mock_platform, tmp_path, mocker)
    result = await agent.run("test-002", "Do something")
    assert result.success is False
    assert result.steps >= 1
```

Run all tests:
```bash
python -m pytest tests/ -v
```

**Checkpoint:** All tests pass before Phase 2.

### Task 7 — Commit Day 2-3 Work

```bash
git add .
git commit -m "fix: OCR working, action saved to db, integration tests added"
git push
```

---

---

# PHASE 2 — Perception Working
### Days 4-5 | Goal: Agent sees and reads screen accurately

---

## Day 4 — Screen Capture Pipeline

### Task 1 — Test Real Screen Capture

```python
python -c "
import sys
sys.path.insert(0, 'src')
from visionnav.perception.capture import ScreenCapture
from visionnav.utils.image import save_screenshot
from pathlib import Path
import numpy as np

cap = ScreenCapture()
arr, meta = cap.capture()
print(f'Screenshot size: {meta[\"width\"]}x{meta[\"height\"]}')
print(f'Array shape: {arr.shape}')

# Save it so you can see it
save_screenshot(arr, Path('test_screenshot.png'))
print('Saved to test_screenshot.png')
"
```

Open `D:\visionnav\test_screenshot.png` — you should see your screen.

---

### Task 2 — Test Full Perception Pipeline

```python
python -c "
import sys
sys.path.insert(0, 'src')
from visionnav.perception.capture import ScreenCapture
from visionnav.perception.ocr import OCREngine
from visionnav.perception.ui_tree import get_ui_tree
from visionnav.perception.fusion import fuse

cap = ScreenCapture()
ocr = OCREngine()

arr, meta = cap.capture()
regions   = ocr.run(arr)
ui        = get_ui_tree()
obs       = fuse(arr, meta, regions, ui)

print('=== SCREEN OBSERVATION ===')
print(f'Screen: {obs.screen_width}x{obs.screen_height}')
print(f'OCR regions found: {len(obs.ocr_regions)}')
print()
print('Text on screen:')
print(obs.to_text_summary())
"
```

This shows exactly what the agent sees when it looks at your screen.

---

### Task 3 — Screenshot Normalization

All screenshots must be same size for consistent model input.

Open `D:\visionnav\src\visionnav\perception\capture.py`

Add this method:

```python
def capture_normalized(
    self,
    target_w: int = 1280,
    target_h: int = 720,
) -> tuple[np.ndarray, dict]:
    """Capture and resize to standard resolution."""
    arr, meta = self.capture()
    if arr.shape[1] != target_w or arr.shape[0] != target_h:
        from visionnav.utils.image import resize_to_target
        arr = resize_to_target(arr, target_w, target_h)
        meta["width"]  = target_w
        meta["height"] = target_h
        meta["resized"] = True
    return arr, meta
```

---

## Day 5 — OCR Quality

### Task 4 — Test OCR On Different Screen Types

Take screenshots of:
```
1. A browser window (text heavy)
2. Desktop with icons
3. A settings page
4. A dark theme window
```

For each run OCR and check:
```
- How many text regions found?
- Is the text accurate?
- Are confidence scores high (>0.8)?
```

### Task 5 — Add OCR Confidence Filter

Open `D:\visionnav\src\visionnav\perception\ocr.py`

Make sure only high confidence text passes:

```python
# Only keep regions with confidence > 0.5
regions = [r for r in regions if r.confidence > 0.5]
```

### Checkpoint — Phase 2 Complete When:

```
✅ Screenshot capture working on any screen
✅ OCR reads text accurately (>80% of visible text)
✅ Observation object contains useful screen information
✅ All perception tests passing
```

Commit:
```bash
git commit -m "feat: perception pipeline working - OCR reads screen accurately"
git push
```

---

---

# PHASE 3 — Data Pipeline
### Days 6-10 | Goal: Training data ready for fine-tuning

---

## Day 6 — Understand The Dataset

### Task 1 — Download GUI-Net-1M Sample

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('Bofeee5675/GUI-Net-1M', split='train', streaming=True)
samples = []
for i, sample in enumerate(ds):
    samples.append(sample)
    if i >= 10:
        break
print(f'Sample keys: {samples[0].keys()}')
print(f'First sample task: {samples[0].get(\"instruction\", \"no instruction\")}')
"
```

### Task 2 — Explore Dataset Structure

For each of 10 samples write down:
```
- What fields does it have?
- Is there a screenshot?
- Is there an action label?
- Is there a task instruction?
- What platform is it from?
```

---

## Day 7 — Data Cleaning Pipeline

### Task 3 — Implement Deduplication

Open `D:\visionnav\data_pipeline\cleaners.py`

Replace stub with real code:

```python
import imagehash
from PIL import Image
from pathlib import Path


def compute_phash(image_path: str) -> str:
    """Compute perceptual hash of image."""
    img = Image.open(image_path).convert("RGB")
    return str(imagehash.phash(img, hash_size=16))


def find_duplicates(image_paths: list[str], threshold: int = 8) -> list[str]:
    """Find duplicate images using perceptual hashing."""
    seen   = {}
    duplicates = []

    for path in image_paths:
        h = compute_phash(path)
        is_dup = False
        for seen_path, seen_h in seen.items():
            if imagehash.hex_to_hash(h) - imagehash.hex_to_hash(seen_h) < threshold:
                duplicates.append(path)
                is_dup = True
                break
        if not is_dup:
            seen[path] = h

    return duplicates


def run_cleaning() -> None:
    print("  Step 1: Finding duplicates...")
    # TODO: Apply to actual dataset
    print("  Step 2: Removing corrupt images...")
    print("  Step 3: Normalizing resolutions...")
    print("  Cleaning complete")
```

---

## Day 8 — Data Formatting

### Task 4 — Implement sharegpt Formatter

Open `D:\visionnav\data_pipeline\formatters.py`

Replace stub with real code:

```python
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are VisionNav, an AI GUI agent. "
    "Think inside <think>...</think> tags. "
    "Output your action inside <action>...</action> tags as JSON."
)


def format_sample(
    image_path: str,
    task: str,
    action_type: str,
    coordinates: list | None = None,
    text: str | None = None,
    reasoning: str = "",
) -> dict:
    """Convert one raw sample to LLaMA-Factory sharegpt format."""

    # Build action JSON
    action = {"type": action_type, "description": f"Performing {action_type}"}
    if coordinates:
        action["coordinates"] = coordinates
    if text:
        action["text"] = text

    # Build assistant response with reasoning
    assistant = ""
    if reasoning:
        assistant += f"<think>\n{reasoning}\n</think>\n"
    assistant += f"<action>\n{json.dumps(action)}\n</action>"

    return {
        "conversations": [
            {"role": "system",    "value": SYSTEM_PROMPT},
            {"role": "user",      "value": f"<image>\nTask: {task}\nWhat is the next action?"},
            {"role": "assistant", "value": assistant},
        ],
        "images": [image_path],
    }


def run_formatting() -> None:
    output_dir = Path("data/instruction_tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("  Formatting samples to sharegpt format...")
    # TODO: Apply to actual cleaned dataset
    print("  Formatting complete")
```

---

## Day 9 — Data Validation

### Task 5 — Implement Real Validator

Open `D:\visionnav\data_pipeline\validators.py`

Update `_validate` function:

```python
def _validate(s: dict) -> None:
    # Check structure
    assert "conversations" in s,    "Missing conversations key"
    assert "images" in s,           "Missing images key"
    assert len(s["conversations"]) == 3, "Need exactly 3 turns: system, user, assistant"
    assert len(s["images"]) >= 1,   "Need at least 1 image"

    # Check roles
    roles = [c["role"] for c in s["conversations"]]
    assert roles == ["system", "user", "assistant"], f"Wrong roles: {roles}"

    # Check assistant has action block
    assistant = s["conversations"][2]["value"]
    assert "<action>" in assistant, "Missing <action> block in assistant response"
    assert "</action>" in assistant, "Missing </action> closing tag"

    # Check action is valid JSON
    import re, json
    match = re.search(r"<action>(.*?)</action>", assistant, re.DOTALL)
    assert match, "Cannot find action content"
    action = json.loads(match.group(1).strip())
    assert "type" in action, "Action missing type field"
```

---

## Day 10 — Full Pipeline Test

### Task 6 — Run Complete Data Pipeline

```bash
python -m data_pipeline.pipeline --stage all
```

### Task 7 — Check Output

```bash
python -c "
import json
from pathlib import Path

files = list(Path('data/instruction_tuning').glob('*.jsonl'))
print(f'Found {len(files)} output files')
for f in files:
    count = sum(1 for _ in open(f))
    print(f'  {f.name}: {count} samples')
"
```

### Checkpoint — Phase 3 Complete When:

```
✅ Can download GUI-Net-1M samples
✅ Deduplication removes duplicates
✅ Formatter converts samples to sharegpt format
✅ Validator confirms all samples are correct
✅ Pipeline runs end to end without errors
```

Commit:
```bash
git commit -m "feat: data pipeline working - samples formatted for training"
git push
```

---

---

# PHASE 4 — Model Fine-Tuning
### Days 11-20 | Goal: Model trained on GUI data

---

## Important — This Phase Requires Google Colab GPU

Setup Colab first using `colab.md` guide.

---

## Day 11 — Setup Training Environment (Colab)

### Task 1 — Open Colab

Follow `colab.md` steps 1-8.

### Task 2 — Install Training Dependencies

```python
!pip install -q \
    llamafactory \
    wandb \
    datasets \
    peft \
    accelerate \
    flash-attn
```

### Task 3 — Verify Training Config

Open `configs/training/sft_3b.yaml` and verify:

```yaml
model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct
finetuning_type: lora
lora_rank: 64
dataset: gui_video_full,wikihow_v3
output_dir: checkpoints/stage1_grounding
num_train_epochs: 3
```

---

## Days 12-13 — Stage 1 Training (Grounding)

### Goal: Model learns to locate UI elements

**Run in Colab:**
```bash
llamafactory-cli train configs/training/sft_3b.yaml
```

**Monitor:**
- Open Weights & Biases dashboard
- Watch training loss go down
- Loss should decrease from ~3.0 to ~0.8

**Stop training when:**
- 3 epochs complete
- OR loss stops decreasing for 500 steps

---

## Day 14 — Evaluate Stage 1

### Run ScreenSpot Evaluation:

```bash
python training/evaluation/eval_screenspot.py \
    --model checkpoints/stage1_grounding \
    --threshold 75.0
```

**Gate:** Must reach 75% before Stage 2.

If below 75%:
- Check training loss curve
- Try training 1 more epoch
- Check data quality

---

## Days 15-16 — Stage 2 Training (Action Prediction)

### Goal: Model learns to predict correct actions

```bash
llamafactory-cli train configs/training/sft_3b_stage2.yaml
```

**Gate:** Action type accuracy >= 85%

---

## Days 17-18 — Stage 3 Training (Planning)

### Goal: Model reasons across multiple steps

```bash
llamafactory-cli train configs/training/sft_3b_stage3.yaml
```

**Gate:**
```
ScreenSpot >= 79.6%
Mind2Web Step SR >= 44%
```

---

## Day 19 — Export Trained Model

```bash
bash training/scripts/export_model.sh checkpoints/stage3_planning
```

This merges LoRA weights into the base model.

---

## Day 20 — Switch From Small Model To Trained Model

Open `D:\visionnav\src\visionnav\settings.py`

Change back:
```python
name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
```

But point to your fine-tuned checkpoint:
```python
name: str = "/content/drive/MyDrive/models/visionnav-3b-finetuned"
```

### Checkpoint — Phase 4 Complete When:

```
✅ Stage 1 complete: ScreenSpot >= 75%
✅ Stage 2 complete: Action accuracy >= 85%
✅ Stage 3 complete: ScreenSpot >= 79.6%
✅ Model exported and loaded successfully
✅ Agent makes correct GUI decisions
```

Commit:
```bash
git commit -m "feat: fine-tuned model integrated - agent makes correct GUI decisions"
git push
```

---

---

# PHASE 5 — Automation Engine
### Days 21-25 | Goal: Agent actually controls the computer

---

## Day 21 — Test Desktop Automation

### Task 1 — Test Basic Click

```python
python -c "
import sys, asyncio
sys.path.insert(0, 'src')
from visionnav.platforms.desktop import DesktopPlatform

async def test():
    p = DesktopPlatform()
    w, h = p.get_screen_size()
    print(f'Screen: {w}x{h}')

    # Click center of screen
    print('Clicking center of screen in 3 seconds...')
    import time
    time.sleep(3)
    result = await p.execute_click(w//2, h//2)
    print(f'Click result: {result}')

asyncio.run(test())
"
```

### Task 2 — Test Keyboard Input

```python
python -c "
import sys, asyncio
sys.path.insert(0, 'src')
from visionnav.platforms.desktop import DesktopPlatform

async def test():
    p = DesktopPlatform()
    print('Opening Run dialog in 3 seconds...')
    import time
    time.sleep(3)
    await p.execute_key('win+r')
    time.sleep(1)
    await p.execute_type('notepad')
    await p.execute_key('enter')
    print('Notepad should be opening...')

asyncio.run(test())
"
```

---

## Day 22 — Test Action Verifier With Real Screen

```python
python -c "
import sys, asyncio
sys.path.insert(0, 'src')
from visionnav.platforms.desktop import DesktopPlatform
from visionnav.actions.verifier import ActionVerifier
from visionnav.actions.schema import Action, ActionType
import time

async def test():
    p = DesktopPlatform()
    v = ActionVerifier()

    # Capture before
    before, _ = await p.capture()
    print('Captured before screenshot')

    # Open notepad (causes screen change)
    print('Opening notepad in 2 seconds...')
    time.sleep(2)
    await p.execute_key('win+r')
    time.sleep(1)
    await p.execute_type('notepad')
    await p.execute_key('enter')
    time.sleep(2)

    # Capture after
    after, _ = await p.capture()
    print('Captured after screenshot')

    # Verify change
    action = Action(type=ActionType.KEY, key='win+r')
    success, ratio = v.verify(before, after, action)
    print(f'Screen changed: {success}')
    print(f'Change ratio: {ratio:.3f}')

asyncio.run(test())
"
```

---

## Days 23-25 — End-to-End Automation Tests

Test these real tasks with trained model:

```
Test 1: "Open Notepad"
        Expected: Notepad window opens

Test 2: "Open Notepad and type Hello World"
        Expected: Notepad opens, text appears

Test 3: "Open Chrome"
        Expected: Chrome browser opens

Test 4: "Take a screenshot and save it"
        Expected: Screenshot saved to file
```

For each test record:
```
- Did agent understand the task?
- Did it find the correct element?
- Did it click in the right place?
- Did it verify the action worked?
```

### Checkpoint — Phase 5 Complete When:

```
✅ Agent can click anywhere on screen accurately
✅ Agent can type text into any field
✅ Agent can open applications
✅ Verifier detects screen changes correctly
✅ 3 out of 4 automation tests pass
```

---

---

# PHASE 6 — Full Agent Loop
### Days 26-30 | Goal: Complete tasks from start to finish

---

## Day 26 — Complex Multi-Step Tasks

Test these complete workflows:

```
Task 1: "Open Chrome, go to google.com, search for AI news"
Steps:  Open Chrome → Wait → Click address bar → Type URL →
        Press Enter → Find search bar → Type query → Press Enter

Task 2: "Open Notepad, type a poem, save the file"
Steps:  Open Notepad → Type poem → Press Ctrl+S →
        Type filename → Press Enter

Task 3: "Take a screenshot and open it in Paint"
Steps:  Press PrintScreen → Open Paint → Paste → Save
```

---

## Day 27 — Error Recovery Testing

Test what happens when agent fails:

```
Scenario 1: Target element not visible
            → Agent should scroll and retry

Scenario 2: Wrong element clicked
            → Agent should detect no change and try again

Scenario 3: Application takes time to open
            → Agent should wait and check again

Scenario 4: Unexpected dialog appears
            → Agent should handle or dismiss it
```

---

## Day 28 — Add Real Memory

Update agent to remember context across steps:

Open `D:\visionnav\src\visionnav\agent\agent.py`

Make history work properly:

```python
# Keep last 10 steps in context
history_dicts = [s.to_history_entry() for s in history[-10:]]
```

---

## Days 29-30 — Internal Benchmark

Run our 100-task benchmark:

```bash
python training/evaluation/eval_visionnav.py \
    --model checkpoints/stage3_planning \
    --platform desktop
```

**Target:** 60% of tasks complete successfully.

### Checkpoint — Phase 6 Complete When:

```
✅ Agent completes 3-step tasks correctly
✅ Error recovery works in 50% of failures
✅ Memory keeps context across 10 steps
✅ Internal benchmark: 60% success rate
```

---

---

# PHASE 7 — API Polish
### Days 31-35 | Goal: Clean professional API

---

## Day 31 — Add WebSocket Streaming

Users can watch agent work in real time:

```python
# In api/v1/tasks.py add:
@router.websocket("/{task_id}/stream")
async def stream_task(websocket: WebSocket, task_id: str):
    await websocket.accept()
    # Send live step updates
    while True:
        steps = await memory.get_recent_steps(task_id, n=1)
        if steps:
            await websocket.send_json(steps[-1].__dict__)
        await asyncio.sleep(1)
```

---

## Day 32 — Add Task Cancellation

```python
@router.delete("/{task_id}")
async def cancel_task(task_id: str) -> dict:
    # Mark task as cancelled in DB
    await memory.mark_task_complete(task_id, False, "Cancelled by user")
    return {"task_id": task_id, "status": "cancelled"}
```

---

## Day 33 — Add Screenshot Endpoint

```python
@router.get("/{task_id}/screenshots")
async def get_screenshots(task_id: str) -> list:
    steps = await memory.get_task_history(task_id)
    return [{"step": s.step_index, "path": s.screenshot_path} for s in steps]
```

---

## Days 34-35 — API Documentation

```bash
# Generate OpenAPI spec
bash scripts/generate_openapi.sh

# View at
http://localhost:8000/docs
```

Make sure every endpoint has:
- Clear description
- Request example
- Response example
- Error codes

---

---

# PHASE 8 — Deployment
### Days 36-40 | Goal: Running in Docker, ready for cloud

---

## Day 36 — Docker Setup

```bash
# Build API image
docker build -f docker/Dockerfile.api -t visionnav-api:latest .

# Test it runs
docker run -p 8000:8000 visionnav-api:latest

# Check health
curl http://localhost:8000/v1/health
```

---

## Day 37 — Docker Compose

```bash
# Start full stack
docker compose -f docker/docker-compose.dev.yml up

# Check all services running
docker compose ps
```

---

## Days 38-39 — Environment Configuration

Make sure these all work:

```bash
# Development
VISIONNAV_ENV=development python -m uvicorn ...

# Production config
VISIONNAV_ENV=production
VISIONNAV_MODEL__BACKEND=vllm
VISIONNAV_DB__URL=postgresql+asyncpg://...
```

---

## Day 40 — Cloud Readiness Check

```
✅ Docker images build cleanly
✅ All config via environment variables
✅ No hardcoded paths or secrets
✅ Health endpoints working
✅ Logs are structured JSON
✅ Ready for AWS ECS deployment
```

---

---

# PHASE 9 — Apps & Future
### Day 41+ | Goal: Real user-facing applications

---

## Desktop App (Electron)

```
Week 1: Basic Electron window
Week 2: Connect to API
Week 3: Live screenshot viewer
Week 4: Task submission UI
Week 5: System tray integration
```

---

## Mobile App (React Native)

```
Week 1: Basic React Native setup
Week 2: Task submission screen
Week 3: Connect to API
Week 4: View task results
Week 5: Android automation
```

---

## Web Dashboard (Next.js)

```
Week 1: Next.js setup
Week 2: Task history page
Week 3: Live task viewer
Week 4: API key management
Week 5: Deploy to Vercel
```

---

---

## Summary Timeline

```
Days 1-3   → Fix & Stabilize         ← YOU ARE HERE
Days 4-5   → Perception Working
Days 6-10  → Data Pipeline
Days 11-20 → Model Fine-Tuning        (Colab GPU)
Days 21-25 → Automation Engine
Days 26-30 → Full Agent Loop
Days 31-35 → API Polish
Days 36-40 → Deployment
Days 41+   → Apps
```

---

## Rules For Every Day

```
1. Start at 8:00 AM
2. Run tests before writing new code
3. Commit to GitHub every evening
4. Never skip a checkpoint
5. If stuck for 30 minutes → ask senior (Claude)
6. Write one thing you learned each day
7. Never paste code you don't understand
```

---

## How To Ask For Help

Always tell me:

```
Phase    : [number and name]
Task     : [what you are working on]
Did      : [what you tried]
Error    : [exact error message]
Think    : [what you think is wrong]
Question : [specific question]
```

---

## What You Will Know When Done

```
✅ Python professional project structure
✅ Virtual environments and package management
✅ FastAPI REST API development
✅ Database design with SQLite/PostgreSQL
✅ Computer vision and OCR
✅ HuggingFace transformers and models
✅ LoRA fine-tuning of large language models
✅ AI agent architecture and design patterns
✅ Docker and containerization
✅ Git and GitHub workflows
✅ Testing (unit, integration, E2E)
✅ Debugging production errors
✅ Cloud deployment readiness
```

**After this project you will be in the top 10% of AI engineers.**
Not because the project is impressive.
Because you built every piece yourself and understand it completely.

---

*See you at 8:00 AM tomorrow. Good night.*
