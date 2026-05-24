# SESSION 4 — The Agent Loop: The Brain of VisionNav

This is the most important session. Every other module exists to serve the agent loop. Understanding this deeply means understanding the entire system.

---

## Part 1 — What a State Machine Is

Before touching any VisionNav code, you need this mental model.

A **state machine** is a system that:
- Exists in exactly one **state** at any moment
- Moves between states based on **events**
- Has a defined set of **transitions** (what causes what)
- Has **terminal states** where it stops

Real world examples:
```
Traffic light:
  States:      RED → GREEN → YELLOW → RED
  Events:      timer expires
  Terminal:    none (loops forever)

Order in a restaurant:
  States:      PLACED → COOKING → READY → DELIVERED → PAID
  Events:      kitchen starts, food done, waiter delivers, customer pays
  Terminal:    PAID (order is done)

VisionNav Agent Task:
  States:      RUNNING → (COMPLETED | FAILED | MAX_STEPS_REACHED)
  Events:      each step result (action succeeded/failed, model says done/fail)
  Terminal:    COMPLETED, FAILED, MAX_STEPS_REACHED
```

Now look at `agent.py`. The `run()` method IS a state machine — it just
is not written explicitly as one. The `for` loop + `if action.type == DONE`
logic IS the transition table.

```python
# This is an implicit state machine
for step_num in range(self._settings.max_steps):

    # ... perceive, reason, act ...

    if action.type == ActionType.DONE:
        # TRANSITION: RUNNING → COMPLETED
        return TaskResult(success=True, ...)

    if action.type == ActionType.FAIL:
        # TRANSITION: RUNNING → FAILED
        return TaskResult(success=False, ...)

# TRANSITION: RUNNING → MAX_STEPS_REACHED
return TaskResult(success=False, error="Maximum steps reached")
```

Why does this matter? Because implicit state machines are hard to test,
extend, and debug. An explicit state machine is clear, testable, and correct.

---

## Part 2 — The Agent Loop Step by Step

Let me trace every single line of execution for one agent step.

```
╔══════════════════════════════════════════════════════════════════╗
║                    ONE AGENT STEP                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. PERCEIVE                                                     ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. capture screenshot → numpy (H,W,3)              │      ║
║     │ b. run OCR on array → list[TextRegion]             │      ║
║     │ c. get_ui_tree → list[dict]                        │      ║
║     │ d. fuse(array, meta, regions, tree) → Observation  │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  2. REASON                                                       ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. planner.get_plan(task) → list[str]              │      ║
║     │ b. memory.get_recent_steps(task_id, n=10)          │      ║
║     │    → list[AgentState] (history)                    │      ║
║     │ c. model.predict_action(                           │      ║
║     │       observation, task, history, plan)            │      ║
║     │    → raw string with <think> and <action>          │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  3. PARSE + SAFETY                                               ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. parse_action(raw_string)                        │      ║
║     │    → Action object OR raises ActionParseError      │      ║
║     │ b. safety.classify(action) → RiskLevel             │      ║
║     │ c. if BLOCKED → replace with FAIL action           │      ║
║     │ d. if HIGH → require confirmation (future)         │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  4. EXECUTE                                                      ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. capture before_screenshot                       │      ║
║     │ b. executor.execute(action, width, height)         │      ║
║     │    → dispatches to pyautogui/ADB/etc.              │      ║
║     │ c. wait for OS to respond                          │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  5. VERIFY                                                       ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. capture after_screenshot                        │      ║
║     │ b. verifier.verify(before, after, action)          │      ║
║     │    → (success: bool, change_ratio: float)          │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  6. RECORD                                                       ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ a. save screenshot to disk                         │      ║
║     │ b. build AgentState dataclass                      │      ║
║     │ c. memory.save_step(task_id, state)                │      ║
║     └────────────────────────────────────────────────────┘      ║
║               ↓                                                  ║
║  7. TRANSITION                                                   ║
║     ┌────────────────────────────────────────────────────┐      ║
║     │ DONE action?     → COMPLETED (return success)      │      ║
║     │ FAIL action?     → FAILED    (return failure)      │      ║
║     │ step == max-1?   → MAX_STEPS (return failure)      │      ║
║     │ otherwise?       → CONTINUE  (go to step 1)        │      ║
║     └────────────────────────────────────────────────────┘      ║
╚══════════════════════════════════════════════════════════════════╝
```

Seven stages. Every stage serves exactly one purpose.
If any stage fails, it is caught and converted to a FAIL action.
The agent never crashes — it always exits gracefully.

---

## Part 3 — How History Works (And Why It Matters)

The agent has no built-in memory between steps.
Every step it starts fresh — it sees the current screen and nothing else.

So how does it know what it already did?

The **history** is passed explicitly to the model every step:

```python
history = await self._memory.get_recent_steps(task_id, n=10)
raw_output = await self._model.predict_action(
    observation,
    task,
    history,      # ← the last 10 steps
    plan,
)
```

The model receives something like:
```
Task: Open Gmail and reply to the most recent email

Previous steps:
  [0] key(win+r) → success → screen changed 0.097
  [1] type(chrome) → success → screen changed 0.019
  [2] key(enter) → success → screen changed 0.41
  [3] click(0.5, 0.055) → success → screen changed 0.15

Current screen shows:
  - 'Gmail' at [0.33, 0.02, 0.44, 0.05]
  - 'Compose' at [0.05, 0.15, 0.18, 0.2]
  - 'John Smith' at [0.18, 0.22, 0.6, 0.27]
  ...

What is the next action?
```

The model sees its own history and reasons from it.
This is why `memory.save_step` on every iteration is critical — without it,
the model has amnesia and cannot make progress on multi-step tasks.

**The 10-step limit:**

We only pass the last 10 steps to control the prompt size.
Each step adds ~100 tokens to the prompt.
10 steps = ~1000 tokens of history.
A 50-step task would need 5000 tokens of history — too much.

In Phase 10 we will add **history compression** — summarize older steps
into a shorter description. But for MVP, 10 steps is the pragmatic limit.

---

## Part 4 — The Prompt Builder: The Most Underrated Component

The prompt builder in `models/prompt.py` is underrated.
It decides exactly what information the model receives.

Information quality directly determines action quality.

```
Better prompt → better model output → better actions → higher task success
```

The current prompt structure:
```
[SYSTEM PROMPT]
You are VisionNav, an AI GUI agent. Think inside <think>...</think>
then output your action inside <action>...</action> as JSON.

[USER MESSAGE]
<image>                             ← the actual screenshot
Task: {task}

Step history:
{formatted_history}

Current screen text:
{observation.to_text_summary()}

Plan:
{formatted_plan}

What is the next action?
```

Every section exists for a reason:

```
<image>                → VLM needs raw visual input
task                   → what we are trying to achieve
formatted_history      → what has already happened
to_text_summary()      → structured text from OCR (more reliable than model vision)
formatted_plan         → high-level steps to guide reasoning
```

The `<think>` tag forces chain-of-thought reasoning before the action.
Research shows models that reason before acting make 30-40% fewer mistakes.

---

## Part 5 — Error Handling in the Agent Loop

The agent loop has four error scenarios. Each is handled differently.

### Scenario 1 — OCR fails completely
```python
try:
    regions = self._ocr.run(arr)
except Exception as exc:
    log.warning("ocr_failed", error=str(exc))
    regions = []   # ← continue with empty OCR, not crash
```
Empty OCR means the model sees the image but no structured text.
It can still reason from the image alone. Degraded but not broken.

### Scenario 2 — Model output is unparseable
```python
try:
    action = parse_action(raw_output)
except ActionParseError as exc:
    log.warning("parse_failed", step=step_num, error=str(exc))
    action = Action(
        type=ActionType.FAIL,
        description=f"Model output unparseable: {exc}",
    )
```
The agent creates a FAIL action and terminates gracefully.
The user receives a proper error, not a Python traceback.

### Scenario 3 — Action execution fails
```python
success = await self._executor.execute(action, w, h)
# success is False if pyautogui raised an exception internally
# The step is saved with action_success=False
# The agent continues to next step (not terminated)
```
A failed execution is not fatal — the agent tries again next step.
After 3 consecutive failures the agent should give up (Phase 10 adds this).

### Scenario 4 — Safety blocks the action
```python
risk = self._safety.classify(action)
if risk == RiskLevel.BLOCKED:
    action = Action(type=ActionType.FAIL, description="Safety blocked")
```
The agent never executes a BLOCKED action — it terminates safely.

---

## Part 6 — Why Background Tasks Matter in the API

When you call `POST /v1/tasks/` — FastAPI returns 202 immediately.
The agent runs in a background task.

```python
async def run_task() -> None:
    agent = await _build_agent()
    await agent.run(task_id, body.instruction)

background_tasks.add_task(run_task)
return TaskResponse(task_id=task_id, status="accepted", ...)
```

Why does this work? Because of async.

```
Timeline:

t=0:   POST /v1/tasks/ arrives
t=0:   FastAPI spawns background coroutine (does NOT run it yet)
t=0:   FastAPI returns 202 response to user
t=0:   User immediately has task_id

t=1ms: Event loop starts running the background coroutine
t=50s: Background coroutine (agent) finishes task
t=50s: memory.mark_task_complete() stores result

t=anytime: User polls GET /v1/tasks/{id} → gets current status
```

Without async, the POST would block for 50 seconds before returning.
The user would think the server crashed.
With async, the user gets a response in < 1ms and polls for progress.

---

## Part 7 — The AgentState Dataclass

Every step is recorded as an `AgentState`. Look at it carefully:

```python
@dataclass
class AgentState:
    step_index:       int
    task_instruction: str        # repeated every step for self-containment
    screenshot_path:  str        # where the screenshot is stored
    ocr_text:         str        # what OCR saw (for quick review)
    action_taken:     Action | None
    action_success:   bool
    reasoning:        str        # what the model was "thinking"
    timestamp:        datetime
    error:            str | None
```

Why is `task_instruction` repeated on every step?

Because each `AgentState` must be **self-contained** — you can look at
any single step and understand everything about it without needing context.
This is critical for debugging and for the history prompt.

Why is `reasoning` stored separately?

Because the `<think>` content is the model's internal reasoning.
It is not shown to the user. It is used for debugging bad decisions.
When the agent does something wrong, the reasoning tells you WHY.

---

## Part 8 — What Production Agent Loops Look Like

Our current loop is MVP. Here is what a production loop adds:

```
MVP Loop (what we have):
  perceive → reason → act → verify → record → check terminal

Production Loop (Phase 10 target):
  perceive
    → fuse_parallel (screenshot + ui_tree concurrently)
    → ocr with caching (same screenshot → skip OCR, reuse)
  reason
    → dynamic context builder (compress history intelligently)
    → world model (predict expected screen BEFORE acting)
    → confidence scoring (is the model sure about this action?)
  act
    → accessibility API first (100% accurate for named elements)
    → coordinate click as fallback (VLM coordinates)
    → retry with different strategy on failure
  verify
    → ScreenDiffAnalyzer (your Assignment 3 — now it pays off)
    → compare expected vs actual change
    → if mismatch → replan
  record
    → async write (don't block on database)
    → screenshot compression (save disk space)
  check terminal
    → DONE / FAIL / MAX_STEPS
    → NEW: ERROR_RECOVERY (retry failed step with different approach)
    → NEW: CONFIDENCE_LOW (ask user for clarification)
```

Your Assignment 3 (`ScreenDiffAnalyzer`) is not just an exercise.
It is a real component that replaces the simple verifier in Phase 10.
This is exactly how professional engineering works: build the component,
prove it works in isolation with tests, then wire it into the system.

---

---

# ASSIGNMENT 4 — The Intelligent Agent Loop

## What You Are Building

A production-grade `AgentLoop` class that separates loop concerns from
`VisionNavAgent`, adds explicit state tracking, implements exponential
backoff retry, and integrates your `ScreenDiffAnalyzer` from Assignment 3.

This is the most advanced assignment so far. It trains:
- State machine design (explicit, testable)
- Exponential backoff (production error handling pattern)
- Separation of concerns (agent vs loop vs policy)
- Integration of multiple modules (everything comes together)
- Advanced async patterns
- Production-grade logging
- Complex class design

---

## The Architecture You Will Build

```
VisionNavAgent
    │
    └── uses → AgentLoop
                    │
                    ├── StepContext     (data for one step)
                    ├── LoopState       (current machine state)
                    ├── RetryPolicy     (when/how to retry)
                    └── StepExecutor    (runs one step)
```

---

## Part A — The Data Structures

Create `src/visionnav/agent/loop.py`:

```python
"""
AgentLoop — Explicit state machine for the VisionNav agent execution loop.

Separates loop control logic from VisionNavAgent, making both
independently testable and replaceable.

Design principles:
  - Explicit state machine (not implicit if/else)
  - Every state transition is logged
  - Retry with exponential backoff on transient failures
  - ScreenDiffAnalyzer integration for intelligent verification
  - Zero side effects in pure functions (classify_transition, etc.)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING

import structlog

from visionnav.actions.schema import Action, ActionType
from visionnav.actions.screen_diff import ChangeType, DiffResult, ScreenDiffAnalyzer
from visionnav.agent.state import AgentState, TaskResult

if TYPE_CHECKING:
    from visionnav.perception.fusion import Observation

log = structlog.get_logger(__name__)


# ─── Loop States ──────────────────────────────────────────────────────────────

class LoopState(Enum):
    """
    Explicit states of the agent execution loop.
    The loop can only be in ONE of these states at any time.
    """
    INITIALIZING    = auto()  # setting up for first step
    PERCEIVING      = auto()  # capturing and processing screen
    REASONING       = auto()  # model deciding next action
    EXECUTING       = auto()  # running the action on OS
    VERIFYING       = auto()  # checking if action worked
    RECORDING       = auto()  # saving step to memory
    RETRYING        = auto()  # action failed, waiting before retry
    COMPLETED       = auto()  # task finished successfully (terminal)
    FAILED          = auto()  # task failed permanently (terminal)
    MAX_STEPS       = auto()  # ran out of steps (terminal)

    @property
    def is_terminal(self) -> bool:
        """Terminal states end the loop."""
        return self in (LoopState.COMPLETED, LoopState.FAILED, LoopState.MAX_STEPS)


# ─── Step Context ─────────────────────────────────────────────────────────────

@dataclass
class StepContext:
    """
    All data gathered and produced during one agent step.
    Immutable once a field is set — enforces correct data flow.

    This is the "working memory" for a single step.
    After the step completes it becomes part of history.
    """
    step_index:       int
    task_instruction: str
    started_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Filled during PERCEIVING
    observation:      Observation | None = None
    before_screenshot: "np.ndarray | None" = None   # type: ignore

    # Filled during REASONING
    raw_model_output: str = ""
    action_planned:   Action | None = None
    reasoning:        str = ""

    # Filled during EXECUTING
    after_screenshot:  "np.ndarray | None" = None   # type: ignore
    execution_success: bool = False

    # Filled during VERIFYING
    diff_result:       DiffResult | None = None

    # Filled on completion
    error:             str | None = None


# ─── Retry Policy ─────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Controls when and how the loop retries failed steps.

    Uses exponential backoff:
      Attempt 1: wait 0.5s
      Attempt 2: wait 1.0s
      Attempt 3: wait 2.0s
      Attempt 4: give up → FAIL

    Why exponential backoff?
    If the OS is slow (loading), waiting longer between retries
    gives it more time to respond. Linear or no backoff wastes retries.
    """
    max_retries:       int   = 3
    base_delay_s:      float = 0.5
    backoff_multiplier: float = 2.0
    max_delay_s:       float = 8.0

    def delay_for_attempt(self, attempt: int) -> float:
        """
        Calculate wait time for the given retry attempt number.

        Args:
            attempt: 1-based retry number (1=first retry, 2=second, ...)

        Returns:
            Seconds to wait before this retry attempt.

        Example:
            attempt=1 → 0.5s
            attempt=2 → 1.0s
            attempt=3 → 2.0s
        """
        # Your implementation here

    def should_retry(self, attempt: int, error: str) -> bool:
        """
        Decide if this failure is worth retrying.

        Retry if:
          - attempt < max_retries
          - error is not permanent (not a safety block, not a parse failure)

        Do NOT retry if:
          - "safety" in error.lower()   ← safety block is permanent
          - "blocked" in error.lower()  ← same
          - attempt >= max_retries      ← exhausted retries

        Args:
            attempt: how many times we have already retried
            error:   the error message from the failed step

        Returns:
            True if we should try again, False if we should FAIL the task
        """
        # Your implementation here
```

---

## Part B — The AgentLoop Class

Continue in `src/visionnav/agent/loop.py`:

```python
class AgentLoop:
    """
    Explicit state machine for agent task execution.

    Responsibilities:
      - Track current loop state
      - Execute state transitions
      - Apply retry policy on failures
      - Integrate ScreenDiffAnalyzer for intelligent verification
      - Emit structured logs on every state transition

    Does NOT:
      - Know how to perceive (delegates to platform + ocr + fusion)
      - Know how to reason (delegates to model)
      - Know how to execute (delegates to executor)
      - Store results (delegates to memory)

    This separation means: if you change the model, the loop is unchanged.
    If you change the executor, the loop is unchanged.
    The loop ORCHESTRATES — it does not IMPLEMENT.
    """

    def __init__(
        self,
        platform,
        ocr_engine,
        model,
        executor,
        memory,
        safety,
        diff_analyzer: ScreenDiffAnalyzer | None = None,
        max_steps: int = 50,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self._platform      = platform
        self._ocr           = ocr_engine
        self._model         = model
        self._executor      = executor
        self._memory        = memory
        self._safety        = safety
        self._diff          = diff_analyzer or ScreenDiffAnalyzer()
        self._max_steps     = max_steps
        self._retry         = retry_policy or RetryPolicy()

        self._state         = LoopState.INITIALIZING
        self._retry_count   = 0

    @property
    def current_state(self) -> LoopState:
        return self._state

    def _transition(self, new_state: LoopState, **log_kwargs) -> None:
        """
        Move to a new state and log the transition.

        Every state change is logged with:
          - old state
          - new state
          - any additional context

        This makes the loop fully observable — you can trace
        exactly what happened and when by reading the logs.
        """
        old = self._state.name
        self._state = new_state
        log.info(
            "loop_transition",
            from_state=old,
            to_state=new_state.name,
            **log_kwargs,
        )

    async def run(
        self,
        task_id: str,
        task_instruction: str,
        history: list[AgentState],
    ) -> TaskResult:
        """
        Run the agent loop until a terminal state is reached.

        This is the main entry point. It:
          1. Iterates up to max_steps
          2. For each step: perceive → reason → execute → verify → record
          3. Applies retry policy on failures
          4. Returns TaskResult when terminal state reached

        Args:
            task_id:         unique identifier for this task run
            task_instruction: natural language task description
            history:         previous steps (for multi-session memory)

        Returns:
            TaskResult with success status, step count, and summary
        """
        self._transition(LoopState.INITIALIZING, task_id=task_id)
        t0       = time.monotonic()
        all_steps: list[AgentState] = list(history)

        for step_index in range(self._max_steps):

            ctx = StepContext(
                step_index=step_index,
                task_instruction=task_instruction,
            )

            # ── Run one complete step ────────────────────────────────
            result = await self._run_step(task_id, ctx, all_steps)
            # ─────────────────────────────────────────────────────────

            if isinstance(result, AgentState):
                all_steps.append(result)
                await self._memory.save_step(task_id, result)

                # Check for terminal conditions
                terminal = self._check_terminal(result, step_index)
                if terminal is not None:
                    elapsed = time.monotonic() - t0
                    return self._build_result(
                        task_id, terminal, all_steps, elapsed
                    )

                # Reset retry counter on successful step
                self._retry_count = 0

            elif isinstance(result, str):
                # result is an error string — apply retry policy
                should_retry = self._retry.should_retry(
                    self._retry_count, result
                )

                if should_retry:
                    self._retry_count += 1
                    delay = self._retry.delay_for_attempt(self._retry_count)
                    self._transition(
                        LoopState.RETRYING,
                        attempt=self._retry_count,
                        delay_s=delay,
                        error=result,
                    )
                    await asyncio.sleep(delay)
                    step_index -= 1    # don't consume a step on retry
                    continue
                else:
                    # Exhausted retries or permanent error
                    elapsed = time.monotonic() - t0
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        steps=step_index + 1,
                        error=f"Retry exhausted: {result}",
                        elapsed_s=elapsed,
                    )

        # Reached max_steps without terminal action
        self._transition(LoopState.MAX_STEPS, max_steps=self._max_steps)
        elapsed = time.monotonic() - t0
        return TaskResult(
            task_id=task_id,
            success=False,
            steps=self._max_steps,
            error="Maximum steps reached without completing the task",
            elapsed_s=elapsed,
        )

    async def _run_step(
        self,
        task_id: str,
        ctx: StepContext,
        history: list[AgentState],
    ) -> AgentState | str:
        """
        Execute one complete step of the agent loop.

        Returns:
            AgentState  → step completed (may or may not have succeeded)
            str         → error message (step failed with exception)

        This method NEVER raises. All exceptions are caught and
        returned as error strings. The loop decides what to do with them.
        """
        try:
            # ── PERCEIVE ────────────────────────────────────────────
            self._transition(LoopState.PERCEIVING, step=ctx.step_index)
            ctx = await self._perceive(ctx)

            # ── REASON ──────────────────────────────────────────────
            self._transition(LoopState.REASONING, step=ctx.step_index)
            ctx = await self._reason(ctx, history)

            # ── EXECUTE ─────────────────────────────────────────────
            self._transition(LoopState.EXECUTING, step=ctx.step_index,
                             action=ctx.action_planned.type if ctx.action_planned else None)
            ctx = await self._execute(ctx)

            # ── VERIFY ──────────────────────────────────────────────
            self._transition(LoopState.VERIFYING, step=ctx.step_index)
            ctx = await self._verify(ctx)

            # ── RECORD ──────────────────────────────────────────────
            self._transition(LoopState.RECORDING, step=ctx.step_index)
            state = self._build_state(task_id, ctx)
            return state

        except Exception as exc:
            log.error(
                "step_exception",
                step=ctx.step_index,
                error=str(exc),
                exc_info=True,
            )
            return str(exc)

    async def _perceive(self, ctx: StepContext) -> StepContext:
        """
        Capture screen, run OCR, build Observation.
        Stores before_screenshot for verification later.
        """
        # Your implementation:
        # 1. Capture screenshot (save as ctx.before_screenshot)
        # 2. Run OCR on screenshot
        # 3. Get UI tree
        # 4. Build Observation via fuse()
        # 5. Store in ctx.observation
        # 6. Return ctx
        ...

    async def _reason(
        self,
        ctx: StepContext,
        history: list[AgentState],
    ) -> StepContext:
        """
        Call model to decide next action.
        Extract reasoning from <think> block.
        Apply safety classification.
        Handle parse failures gracefully.
        """
        # Your implementation:
        # 1. Get plan from planner (use a simple one or just pass empty list)
        # 2. Call model.predict_action(ctx.observation, task, history, plan)
        # 3. Store raw output in ctx.raw_model_output
        # 4. Extract reasoning from <think>...</think> block
        #    (use regex: re.search(r'<think>(.*?)</think>', raw, re.DOTALL))
        # 5. Parse action: try parse_action(raw) → catch ActionParseError
        #    On error: ctx.action_planned = Action(type=FAIL, description=error)
        # 6. Check safety: if BLOCKED → replace with FAIL action
        # 7. Store in ctx.action_planned, ctx.reasoning
        # 8. Return ctx
        ...

    async def _execute(self, ctx: StepContext) -> StepContext:
        """
        Execute the planned action on the OS.
        Capture after_screenshot for verification.
        """
        # Your implementation:
        # 1. Get screen dimensions from platform
        # 2. Call executor.execute(ctx.action_planned, width, height)
        #    Store result in ctx.execution_success
        # 3. Capture after screenshot (ctx.after_screenshot)
        # 4. Return ctx
        ...

    async def _verify(self, ctx: StepContext) -> StepContext:
        """
        Use ScreenDiffAnalyzer to verify the action had effect.
        Stores full DiffResult (not just True/False).
        """
        # Your implementation:
        # 1. If before_screenshot or after_screenshot is None → skip
        # 2. Call self._diff.analyze(before, after) → DiffResult
        # 3. Store in ctx.diff_result
        # 4. Log the diff result (change_type, change_ratio, summary)
        # 5. Return ctx
        ...

    def _check_terminal(
        self,
        state: AgentState,
        step_index: int,
    ) -> LoopState | None:
        """
        Pure function — no side effects, no async.
        Returns the terminal LoopState if task is done, else None.

        Why pure? Because pure functions are easy to test.
        You can test every edge case without setting up the full loop.
        """
        if state.action_taken is None:
            return None
        if state.action_taken.type == ActionType.DONE:
            return LoopState.COMPLETED
        if state.action_taken.type == ActionType.FAIL:
            return LoopState.FAILED
        return None

    def _build_state(self, task_id: str, ctx: StepContext) -> AgentState:
        """Build an AgentState from a completed StepContext."""
        # Your implementation:
        # Pull fields from ctx into AgentState
        # screenshot_path: you can use "" for now
        # ocr_text: join OCR region texts from observation
        ...

    def _build_result(
        self,
        task_id: str,
        terminal: LoopState,
        all_steps: list[AgentState],
        elapsed_s: float,
    ) -> TaskResult:
        """Build a TaskResult from the terminal state."""
        success = terminal == LoopState.COMPLETED
        self._transition(terminal, steps=len(all_steps), elapsed_s=round(elapsed_s, 2))

        last = all_steps[-1] if all_steps else None
        return TaskResult(
            task_id  = task_id,
            success  = success,
            steps    = len(all_steps),
            summary  = last.action_taken.description if last and last.action_taken else "",
            error    = None if success else "Task failed",
            elapsed_s= elapsed_s,
        )
```

---

## Part C — Wire Into the Agent

Open `src/visionnav/agent/agent.py`.

Add an `AgentLoop` instance as an alternative execution path:

```python
async def run_with_loop(
    self,
    task_id: str,
    instruction: str,
) -> TaskResult:
    """
    Run the task using the new explicit AgentLoop.
    Replaces the implicit state machine in run() with an
    explicit, testable, extensible state machine.

    This is the future of VisionNavAgent execution.
    """
    from visionnav.agent.loop import AgentLoop, RetryPolicy

    loop = AgentLoop(
        platform     = self._platform,
        ocr_engine   = self._ocr,
        model        = self._model,
        executor     = self._executor,
        memory       = self._memory,
        safety       = self._safety,
        max_steps    = self._settings.max_steps,
        retry_policy = RetryPolicy(max_retries=3, base_delay_s=0.5),
    )

    await self._memory.save_task(task_id, instruction)

    history = await self._memory.get_recent_steps(task_id, n=10)

    return await loop.run(task_id, instruction, history)
```

---

## Part D — Write Tests

Create `tests/unit/test_agent_loop.py`:

```python
"""
Tests for AgentLoop — the explicit state machine.

Testing strategy:
  - Test each pure function in isolation first
  - Then test state transitions with mocks
  - Then test retry logic with a deliberately failing model
  - Never test implementation details — only observable behavior
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from visionnav.agent.loop import (
    AgentLoop, LoopState, RetryPolicy, StepContext
)
from visionnav.actions.schema import Action, ActionType
from visionnav.agent.state import AgentState


# ── RetryPolicy Tests (pure functions — no mocks needed) ──────────────────────

class TestRetryPolicy:

    def test_delay_increases_exponentially(self):
        policy = RetryPolicy(base_delay_s=0.5, backoff_multiplier=2.0)
        d1 = policy.delay_for_attempt(1)
        d2 = policy.delay_for_attempt(2)
        d3 = policy.delay_for_attempt(3)
        assert d2 == pytest.approx(d1 * 2.0)
        assert d3 == pytest.approx(d2 * 2.0)

    def test_delay_never_exceeds_max(self):
        policy = RetryPolicy(
            base_delay_s=1.0,
            backoff_multiplier=10.0,
            max_delay_s=5.0,
        )
        assert policy.delay_for_attempt(10) <= 5.0

    def test_should_retry_when_attempts_remain(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(0, "connection error") is True
        assert policy.should_retry(1, "timeout") is True
        assert policy.should_retry(2, "unknown") is True

    def test_should_not_retry_when_exhausted(self):
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(3, "timeout") is False

    def test_should_not_retry_safety_block(self):
        policy = RetryPolicy(max_retries=10)
        assert policy.should_retry(0, "safety blocked this action") is False
        assert policy.should_retry(0, "action blocked by classifier") is False

    def test_delay_attempt_1_equals_base(self):
        policy = RetryPolicy(base_delay_s=0.5)
        assert policy.delay_for_attempt(1) == pytest.approx(0.5)


# ── LoopState Tests ───────────────────────────────────────────────────────────

class TestLoopState:

    def test_terminal_states_are_terminal(self):
        assert LoopState.COMPLETED.is_terminal is True
        assert LoopState.FAILED.is_terminal is True
        assert LoopState.MAX_STEPS.is_terminal is True

    def test_non_terminal_states_are_not_terminal(self):
        assert LoopState.PERCEIVING.is_terminal is False
        assert LoopState.REASONING.is_terminal is False
        assert LoopState.EXECUTING.is_terminal is False
        assert LoopState.RETRYING.is_terminal is False


# ── _check_terminal Tests (pure function) ─────────────────────────────────────

class TestCheckTerminal:

    def _make_loop(self):
        """Create a minimal AgentLoop for testing pure methods."""
        return AgentLoop(
            platform=MagicMock(), ocr_engine=MagicMock(),
            model=MagicMock(), executor=MagicMock(),
            memory=MagicMock(), safety=MagicMock(),
        )

    def _make_state(self, action_type: ActionType) -> AgentState:
        from datetime import datetime, timezone
        return AgentState(
            step_index=0,
            task_instruction="test",
            screenshot_path="",
            ocr_text="",
            action_taken=Action(type=action_type, description="test"),
            action_success=True,
            reasoning="",
            timestamp=datetime.now(timezone.utc),
            error=None,
        )

    def test_done_action_returns_completed(self):
        loop  = self._make_loop()
        state = self._make_state(ActionType.DONE)
        assert loop._check_terminal(state, 0) == LoopState.COMPLETED

    def test_fail_action_returns_failed(self):
        loop  = self._make_loop()
        state = self._make_state(ActionType.FAIL)
        assert loop._check_terminal(state, 0) == LoopState.FAILED

    def test_click_action_returns_none(self):
        loop  = self._make_loop()
        state = self._make_state(ActionType.CLICK)
        assert loop._check_terminal(state, 0) is None

    def test_no_action_returns_none(self):
        from datetime import datetime, timezone
        loop  = self._make_loop()
        state = AgentState(
            step_index=0, task_instruction="test",
            screenshot_path="", ocr_text="",
            action_taken=None,
            action_success=False, reasoning="",
            timestamp=datetime.now(timezone.utc), error=None,
        )
        assert loop._check_terminal(state, 0) is None


# ── Integration Tests with Mocked Components ─────────────────────────────────

class TestAgentLoopIntegration:

    def _make_mock_platform(self):
        import numpy as np
        platform = AsyncMock()
        platform.capture.return_value = (
            np.zeros((100, 100, 3), dtype=np.uint8),
            {"width": 100, "height": 100, "monitor": 1},
        )
        platform.get_ui_tree.return_value   = []
        platform.get_screen_size            = lambda: (100, 100)
        platform.execute_click.return_value = True
        platform.execute_type.return_value  = True
        platform.execute_key.return_value   = True
        return platform

    def _make_mock_model(self, output: str):
        model = AsyncMock()
        model.predict_action.return_value = output
        return model

    def _make_mock_memory(self):
        memory = AsyncMock()
        memory.get_recent_steps.return_value = []
        memory.save_task.return_value        = None
        memory.save_step.return_value        = None
        memory.mark_task_complete.return_value = None
        return memory

    @pytest.mark.asyncio
    async def test_done_on_first_step_succeeds(self):
        """Loop completes when model returns DONE immediately."""
        loop = AgentLoop(
            platform   = self._make_mock_platform(),
            ocr_engine = MagicMock(run=lambda x: []),
            model      = self._make_mock_model(
                '<think>Task done</think>'
                '<action>{"type":"done","description":"Completed"}</action>'
            ),
            executor   = AsyncMock(execute=AsyncMock(return_value=True)),
            memory     = self._make_mock_memory(),
            safety     = MagicMock(classify=MagicMock(return_value=0)),
            max_steps  = 5,
        )
        result = await loop.run("task-1", "Do something", [])
        assert result.success is True
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_fail_action_terminates_loop(self):
        """Loop terminates with failure when model returns FAIL."""
        loop = AgentLoop(
            platform   = self._make_mock_platform(),
            ocr_engine = MagicMock(run=lambda x: []),
            model      = self._make_mock_model(
                '<action>{"type":"fail","description":"Cannot do this"}</action>'
            ),
            executor   = AsyncMock(execute=AsyncMock(return_value=True)),
            memory     = self._make_mock_memory(),
            safety     = MagicMock(classify=MagicMock(return_value=0)),
            max_steps  = 5,
        )
        result = await loop.run("task-2", "Impossible task", [])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_max_steps_terminates_loop(self):
        """Loop stops after max_steps if never completed."""
        loop = AgentLoop(
            platform   = self._make_mock_platform(),
            ocr_engine = MagicMock(run=lambda x: []),
            model      = self._make_mock_model(
                '<action>{"type":"click","coordinates":[0.5,0.5]}</action>'
            ),
            executor   = AsyncMock(execute=AsyncMock(return_value=True)),
            memory     = self._make_mock_memory(),
            safety     = MagicMock(classify=MagicMock(return_value=0)),
            max_steps  = 3,
        )
        result = await loop.run("task-3", "Click forever", [])
        assert result.success is False
        assert result.steps == 3

    @pytest.mark.asyncio
    async def test_retry_on_execution_failure(self):
        """Loop retries when execution fails transiently."""
        # First call fails, second succeeds with DONE
        call_count = {"n": 0}

        async def flaky_model(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("GPU memory spike")
            return '<action>{"type":"done","description":"OK"}</action>'

        model = MagicMock()
        model.predict_action = flaky_model

        loop = AgentLoop(
            platform     = self._make_mock_platform(),
            ocr_engine   = MagicMock(run=lambda x: []),
            model        = model,
            executor     = AsyncMock(execute=AsyncMock(return_value=True)),
            memory       = self._make_mock_memory(),
            safety       = MagicMock(classify=MagicMock(return_value=0)),
            max_steps    = 10,
            retry_policy = RetryPolicy(
                max_retries=3,
                base_delay_s=0.01,   # fast in tests
            ),
        )
        result = await loop.run("task-4", "Retry test", [])
        assert result.success is True
        assert call_count["n"] == 2    # called twice: once failed, once succeeded

    @pytest.mark.asyncio
    async def test_initial_state_is_initializing(self):
        """Loop starts in INITIALIZING state."""
        loop = AgentLoop(
            platform=self._make_mock_platform(),
            ocr_engine=MagicMock(run=lambda x: []),
            model=self._make_mock_model(
                '<action>{"type":"done","description":"OK"}</action>'
            ),
            executor=AsyncMock(execute=AsyncMock(return_value=True)),
            memory=self._make_mock_memory(),
            safety=MagicMock(classify=MagicMock(return_value=0)),
        )
        assert loop.current_state == LoopState.INITIALIZING
```

---

## Success Criteria

```bash
python -m pytest tests/unit/test_agent_loop.py -v
```

All tests pass. Then:

```bash
python -m pytest tests/unit/ tests/integration/ -v \
  --ignore=tests/integration/test_platform_capture.py
```

All existing tests still pass.

---

## What I Will Review

```
State Machine Design:
  □ Is LoopState.is_terminal correctly implemented?
  □ Does _transition log every state change with structured fields?
  □ Is _check_terminal truly pure (no side effects, no async)?

RetryPolicy:
  □ Is delay_for_attempt mathematically correct exponential backoff?
  □ Does max_delay_s cap work correctly?
  □ Does should_retry correctly identify permanent errors?

Integration:
  □ Does _perceive correctly store before_screenshot?
  □ Does _verify use ScreenDiffAnalyzer (Assignment 3 integration)?
  □ Does _reason handle ActionParseError gracefully?
  □ Does _run_step NEVER raise (all exceptions caught and returned)?

Tests:
  □ Does test_retry_on_execution_failure verify call_count == 2?
  □ Is RetryPolicy tested with pure function calls (no mocks needed)?
  □ Does test_max_steps_terminates_loop verify steps == max_steps?
```

---

## One Principle To Carry Forward

> **Explicit is better than implicit.**

The original `agent.py` loop is implicit — state transitions are buried in
`if/elif` chains that you have to read carefully to understand.

The `AgentLoop` is explicit — every state is named, every transition is logged,
every terminal condition is a clear function call.

Six months from now when something goes wrong in production, you will read the
logs and see exactly:
```
loop_transition  from=EXECUTING  to=VERIFYING   step=7
loop_transition  from=VERIFYING  to=RETRYING     attempt=1  delay_s=0.5
loop_transition  from=RETRYING   to=PERCEIVING   step=7
loop_transition  from=EXECUTING  to=VERIFYING   step=7
loop_transition  from=VERIFYING  to=RECORDING    step=7
loop_transition  from=RECORDING  to=COMPLETED    steps=8
```

That is the difference between a system you can operate and a system you can only pray works.

Come back with **"Assignments 2, 3, and 4 submitted"** when all three are ready. I will review all three and then we begin **Phase 16 — Dataset Factory**, which is where the real competitive advantage of VisionNav is built.