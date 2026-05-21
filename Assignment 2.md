## Assignment 1 — ✅ APPROVED

Well done. You fixed all three issues, tests pass, and you documented the warning instead of ignoring it. That last part — documenting known issues — is a senior engineering habit. Approved.

---

---

# SESSION 2 — Async Programming and Why Agents Need It

Before the assignment, you must understand this concept deeply.
Do not skip the explanation. The assignment will not make sense without it.

---

## Why Async Exists — The Restaurant Problem

Imagine a restaurant with one waiter.

**Synchronous waiter (blocking):**
```
Table 1 orders → waiter goes to kitchen → stands and waits for food
→ brings food → goes to Table 2 → takes order → goes to kitchen
→ stands and waits again → brings food → goes to Table 3...
```

While the waiter waits at the kitchen, Tables 2 and 3 are ignored.
One waiter can only serve one table at a time.

**Asynchronous waiter (non-blocking):**
```
Table 1 orders → waiter tells kitchen → goes to Table 2
Table 2 orders → waiter tells kitchen → goes to Table 3
Table 3 orders → waiter tells kitchen
Kitchen calls: "Table 1 ready!" → waiter picks up and delivers
Kitchen calls: "Table 2 ready!" → waiter picks up and delivers
```

Same one waiter. Three tables served simultaneously.
The waiter never stands idle waiting — they always do the next useful thing.

**In Python terms:**
```
Synchronous:
  response = requests.get("http://api.com")   # BLOCKS — thread freezes here
  # Nothing else runs until response arrives (could be 2 seconds)
  print(response.json())

Asynchronous:
  response = await httpx.get("http://api.com")  # SUSPENDS — other code runs
  # While waiting for response, Python runs other coroutines
  print(response.json())
```

---

## What Actually Happens Under the Hood

Python's async system has one component called the **Event Loop**.
Think of it as the restaurant manager who keeps track of everything.

```
Event Loop Manager:
  "I have 3 tasks waiting:
    Task A: waiting for HTTP response from vLLM server
    Task B: waiting for screenshot to be captured
    Task C: waiting for database write to finish

  While Task A waits → run Task B
  While Task B waits → run Task C
  Task C finished → mark complete
  Task A response arrived → resume Task A
  Task B screenshot done → resume Task B"
```

The critical insight:
```
Async does NOT make your code faster for CPU work.
Async makes your code faster for WAITING work (I/O).

CPU work:  calculating, processing images, running model inference
I/O work:  HTTP requests, disk reads, database queries, screenshot capture

Async only helps I/O work.
```

---

## The Three Keywords You Must Master

### 1. `async def` — This function can be paused

```python
# Normal function — runs start to finish, blocks everything
def get_data():
    return requests.get("http://api.com").json()

# Async function — can be paused while waiting
async def get_data():
    return await httpx.get("http://api.com").json()
```

Declaring `async def` does NOT make the function run asynchronously.
It makes it a **coroutine** — a function that CAN be paused.

### 2. `await` — Pause here and let others run

```python
async def agent_step():
    screenshot = await platform.capture()    # pause: wait for capture
    # While capture happens, event loop runs other coroutines
    ocr_result = await ocr_engine.run(screenshot)  # pause: wait for OCR
    action = await model.predict(screenshot)  # pause: wait for VLM
    await executor.execute(action)            # pause: wait for mouse click
```

`await` can ONLY be used inside `async def` functions.
`await` can ONLY be used with objects that support it (coroutines, futures).

### 3. `asyncio.run()` — Start the event loop

```python
async def main():
    result = await agent.run("Open Notepad")
    print(result)

# This starts the event loop and runs main() inside it
asyncio.run(main())
```

---

## Why VisionNav's Agent Loop Is Async

Look at what the agent does in one step:

```
1. Capture screenshot    → calls OS screenshot API    → I/O wait
2. Run OCR               → calls Tesseract subprocess  → I/O wait
3. Get UI tree           → calls accessibility API     → I/O wait
4. Call VLM model        → HTTP request to vLLM        → I/O wait (LONG)
5. Execute action        → calls pyautogui             → I/O wait
6. Save to database      → SQLite write                → I/O wait
```

Every single step is I/O work. The agent spends most of its time WAITING.

If these were synchronous — only ONE thing could happen at a time.
With async — while the VLM thinks (1-2 seconds), the database save can happen.
While the screenshot is capturing, other coroutines can run.

**The deeper reason for async in VisionNav:**

Our API serves MULTIPLE users simultaneously. When User A submits a task
and the agent is waiting for the VLM response — User B's request should not
be stuck waiting. Async allows FastAPI to handle User B's request while
User A's agent waits for the VLM.

```
Without async (FastAPI with sync endpoints):
  User A submits task → VLM thinking (3 seconds) → User B blocked
  
With async (our system):
  User A submits task → VLM thinking → FastAPI handles User B
  → VLM responds → User A's agent continues
```

---

## The Trap That Kills Async Performance

This is the mistake 90% of junior engineers make:

```python
# WRONG — blocking inside async function
async def predict_action(self, observation):
    import time
    time.sleep(2)        # ← BLOCKS THE ENTIRE EVENT LOOP
    return "done"

# WRONG — synchronous I/O inside async function
async def save_screenshot(self, path):
    with open(path, "wb") as f:    # ← blocks on disk I/O
        f.write(data)
```

When you block inside an async function, you block the ENTIRE event loop.
Every other coroutine freezes. You defeated the purpose of async.

```python
# CORRECT — use asyncio.sleep for delays
async def wait_for_load(self, seconds: float):
    await asyncio.sleep(seconds)   # ← yields control, others run

# CORRECT — use run_in_executor for blocking code you can't avoid
async def capture(self):
    loop = asyncio.get_event_loop()
    arr, meta = await loop.run_in_executor(None, self._sync_capture)
    return arr, meta
```

`run_in_executor` runs blocking code in a separate thread pool,
so the event loop is not blocked.

Look at our `DesktopPlatform`:
```python
async def capture(self) -> tuple[np.ndarray, dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self._capture.capture)
```

`mss.capture()` is synchronous (it calls OS APIs directly).
We wrap it in `run_in_executor` so it runs in a thread without blocking
the event loop. This is the correct pattern for any synchronous library.

---

## Async Patterns Used in VisionNav

### Pattern 1 — Sequential async (most common in agent loop)
```python
async def agent_step(self):
    screenshot = await self.platform.capture()   # wait for this
    ocr        = self.ocr.run(screenshot)        # then this (sync — OCR is CPU)
    action     = await self.model.predict(...)   # then wait for this
    await self.executor.execute(action)          # then wait for this
```

### Pattern 2 — Concurrent async (parallel tasks)
```python
async def gather_perception(self):
    screenshot, ui_tree = await asyncio.gather(
        self.platform.capture(),         # these two run simultaneously
        self.platform.get_ui_tree(),     # no need to wait for one before other
    )
    return screenshot, ui_tree
```

`asyncio.gather` runs multiple coroutines concurrently.
If capture takes 50ms and ui_tree takes 30ms — together they take 50ms not 80ms.

### Pattern 3 — Async with timeout
```python
async def predict_with_timeout(self, observation):
    try:
        result = await asyncio.wait_for(
            self.model.predict_action(observation, ...),
            timeout=30.0,    # if model takes > 30 seconds → cancel
        )
        return result
    except asyncio.TimeoutError:
        return self._make_fail_action("Model timeout")
```

This is critical for production — you never want an agent stuck forever
waiting for a model that crashed.

---

---

# ASSIGNMENT 2 — The Async Architect

## What You Are Building

A **parallel perception system** that captures screenshot and UI tree
simultaneously instead of sequentially, and a **timeout-protected model caller**
that never hangs.

This assignment trains:
- Async/await fundamentals
- `asyncio.gather` for concurrent execution
- `asyncio.wait_for` for timeouts
- `run_in_executor` for blocking code
- Understanding where async helps and where it does not
- Writing clean async code

---

## Part A — Parallel Perception

Open `src/visionnav/perception/fusion.py`.

Add a new async function called `fuse_parallel`:

```python
async def fuse_parallel(
    platform,             # PlatformAdapter instance
    ocr_engine,           # OCREngine instance
    meta_override: dict | None = None,
) -> Observation:
    """
    Capture screenshot and UI tree in PARALLEL using asyncio.gather.
    Then run OCR on the captured screenshot.

    Why parallel?
    - Screenshot capture: ~50ms
    - UI tree extraction: ~30ms
    - Sequential total: 80ms
    - Parallel total:   ~50ms (limited by the slower one)

    This is a 37% speed improvement on every single agent step.
    For a 50-step task, that saves ~1.5 seconds total.
    """
    # Your implementation here
    # Step 1: Run capture and get_ui_tree concurrently with asyncio.gather
    # Step 2: Run OCR on the screenshot (OCR is synchronous CPU work)
    # Step 3: Call fuse() to combine everything into Observation
    # Step 4: Return the Observation
```

**Rules:**
- `screenshot capture` and `ui_tree` must run concurrently with `asyncio.gather`
- OCR runs after screenshot is available (it needs the image)
- Use the existing `fuse()` function for the final step
- Add proper type hints on everything
- Add a docstring explaining what it does and why

---

## Part B — Timeout-Protected Model Caller

Open `src/visionnav/models/local.py`.

Add a method called `predict_action_safe`:

```python
async def predict_action_safe(
    self,
    observation: Observation,
    task: str,
    history: list[dict],
    plan: list[str],
    timeout_seconds: float = 30.0,
) -> str:
    """
    Calls predict_action with a timeout.
    If the model takes longer than timeout_seconds:
      - Logs a warning with the timeout duration
      - Returns a valid FAIL action string so the agent can handle it

    Why this matters:
    If the model hangs (GPU OOM, network issue, deadlock):
    Without timeout: agent waits forever, user never gets response
    With timeout:    agent gets FAIL action, reports error, user informed

    Returns:
        Raw model output string (valid even on timeout — returns FAIL action)
    """
    # Your implementation here
    # Use asyncio.wait_for
    # On TimeoutError: log warning, return this exact string:
    # '<action>{"type":"fail","description":"Model timeout after Xs"}</action>'
```

**Rules:**
- Must use `asyncio.wait_for`
- On timeout must return a valid parseable action string (not raise)
- The returned string on timeout must parse without error via `parse_action()`
- Log the timeout with structured logging (include timeout duration)
- Proper type hints

---

## Part C — Benchmark: Sequential vs Parallel

Create a new file: `scripts/benchmark_perception.py`

```python
"""
Benchmark: Sequential vs Parallel perception.

Measures real time difference between:
  - Old approach: capture then get_ui_tree (sequential)
  - New approach: asyncio.gather (parallel)

Run with:
    python scripts/benchmark_perception.py
"""
import asyncio
import time
import sys
sys.path.insert(0, "src")

# Your implementation:
# 1. Run sequential perception 10 times, measure average time
# 2. Run parallel perception 10 times, measure average time
# 3. Print comparison table showing:
#    - Sequential average: X ms
#    - Parallel average:   Y ms
#    - Improvement:        Z% faster
```

The output must look like this:
```
Perception Benchmark (10 runs each)
════════════════════════════════════════
Sequential:  82.3 ms average
Parallel:    51.7 ms average
Improvement: 37.2% faster

Step breakdown:
  Capture:  48.1 ms
  UI Tree:  31.4 ms
  OCR:      15.2 ms
```

---

## Part D — Write Tests

Create `tests/unit/test_async_perception.py`:

```python
# Test 1: fuse_parallel returns valid Observation
@pytest.mark.asyncio
async def test_fuse_parallel_returns_observation(mock_platform):
    # Use mock_platform from conftest
    # Call fuse_parallel
    # Assert result is Observation with correct fields

# Test 2: fuse_parallel is actually faster than sequential
# (this is a real benchmark test)
@pytest.mark.asyncio
async def test_fuse_parallel_faster_than_sequential(mock_platform):
    # Run sequential 5 times → measure average
    # Run parallel 5 times → measure average
    # Assert parallel is not SLOWER (allow ±10ms tolerance since mocks are fast)

# Test 3: timeout returns parseable FAIL action
@pytest.mark.asyncio
async def test_model_timeout_returns_fail_action():
    # Create a model backend where predict_action sleeps for 5 seconds
    # Call predict_action_safe with timeout=0.1
    # Assert result contains <action> block
    # Assert parse_action(result) gives ActionType.FAIL

# Test 4: timeout is logged
@pytest.mark.asyncio
async def test_model_timeout_is_logged(caplog):
    # Same as Test 3 but also verify the warning was logged
    # Use caplog pytest fixture to capture log output
```

---

## Important Rules For This Assignment

**Rule 1:** You cannot use `time.sleep()` anywhere. Only `asyncio.sleep()`.

**Rule 2:** `fuse_parallel` must demonstrate a real speedup. If your mock platform returns instantly (which mocks do), you need to add realistic delays to your benchmark mock. Read how `asyncio.sleep` can simulate real latency.

**Rule 3:** The timeout test MUST use a slow model. Create one like this:
```python
class SlowMockModel:
    async def predict_action(self, *args, **kwargs) -> str:
        await asyncio.sleep(10)   # simulates slow/hung model
        return "this should never be reached in timeout test"
```

**Rule 4:** Every async function must have `async def`. Every call to an async function must use `await`.

**Rule 5:** No bare `except:` clauses. Always catch specific exceptions.

---

## What I Will Review

```
Correctness:
  □ Does fuse_parallel actually run gather concurrently?
  □ Does predict_action_safe actually timeout and return FAIL?
  □ Does the returned timeout string parse without error?

Async understanding:
  □ Is OCR correctly placed AFTER screenshot (it needs the image)?
  □ Is run_in_executor used for any blocking calls you added?
  □ Are there any time.sleep() calls? (violation)

Test quality:
  □ Does test_fuse_parallel use mock_platform from conftest?
  □ Does test_model_timeout actually test a SLOW model?
  □ Does the timeout test verify the FAIL action is parseable?

Code quality:
  □ Type hints on every function
  □ Docstrings explaining WHY not just WHAT
  □ Structured logging for timeout event
  □ Clean variable names
```

---

## One Concept To Carry Into This Assignment

```
asyncio.gather(coro_a(), coro_b())
```

This means: start both coroutines, let them run concurrently,
wait until BOTH finish, return both results.

It is not magic. It only works if the coroutines actually yield control
(i.e., they contain `await` statements that give the event loop a chance to switch).

If you `await` a function that secretly calls `time.sleep()` — no concurrency.
If you `await` a function that calls `asyncio.sleep()` — true concurrency.

This is why mocking matters in async testing.
Your mocks must use `asyncio.sleep` not `time.sleep`.

---

Start the assignment. When you finish, come back with:

1. The `fuse_parallel` function
2. The `predict_action_safe` method
3. The benchmark output (copy from terminal)
4. All tests passing

Say **"Assignment 2 submitted"** when ready.