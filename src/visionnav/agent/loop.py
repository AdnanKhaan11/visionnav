"""
AgentLoop — Explicit state machine for the VisionNav agent execution loop.

Separates loop control logic from VisionNavAgent, making both
independently testable and replaceable.

Design principles:
  - Explicit state machine (not implicit if/else)
  - Every state transition is logged
  - Retry with exponential backoff on transient failures
  - ScreenDiffAnalyzer integration for intelligent verification
  - Zero side effects in pure functions (_check_terminal, etc.)
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING

import structlog

from visionnav.actions.parser import parse_action, ActionParseError
from visionnav.actions.schema import Action, ActionType
from visionnav.actions.screen_diff import ScreenDiffAnalyzer, DiffResult
from visionnav.agent.state import AgentState, TaskResult
from visionnav.perception.fusion import fuse

if TYPE_CHECKING:
    import numpy as np
    from visionnav.perception.fusion import Observation

log = structlog.get_logger(__name__)


# ─── Loop States ──────────────────────────────────────────────────────────────


class LoopState(Enum):
    """
    Every possible state the agent loop can be in.
    The loop is always in exactly ONE state at any moment.

    Think of it like a traffic light:
      A traffic light is always in exactly one state: RED, GREEN, or YELLOW.
      It never shows two colors at once.
      Our loop works the same way.
    """

    # auto() automatically assigns unique values to each state, starting from 1, we write auto because we don't care about the actual values, just that they are unique.
    # what is enum? enum is a way to define a set of named constants that are related named constants simple means that each state has a unique name (PERCEIVING, REASONING) and a unique value (1, 2, 3).
    INITIALIZING = auto()  # setting up — before first step starts
    PERCEIVING = auto()  # capturing screenshot and reading text
    REASONING = auto()  # model deciding what to do next
    EXECUTING = auto()  # physically doing the action (mouse, keyboard)
    VERIFYING = auto()  # checking if the action worked
    RECORDING = auto()  # saving this step to database
    RETRYING = auto()  # something failed — waiting before trying again
    COMPLETED = auto()  # task finished successfully ← terminal
    FAILED = auto()  # task failed permanently   ← terminal
    MAX_STEPS = auto()  # ran out of steps          ← terminal

    @property  # property means  This function behaves like a variable."
    def is_terminal(self) -> bool:
        """
        Terminal states end the loop permanently.
        Once reached, the loop never runs another step.
        """
        return self in (LoopState.COMPLETED, LoopState.FAILED, LoopState.MAX_STEPS)


# ─── Step Context ─────────────────────────────────────────────────────────────


@dataclass
class StepContext:
    """
    A container that holds ALL data for one single agent step.

    Think of it like a clipboard that gets passed through an assembly line:
      Station 1 (PERCEIVE)  → writes screenshot and OCR text
      Station 2 (REASON)    → writes planned action and reasoning
      Station 3 (EXECUTE)   → writes success result and after screenshot
      Station 4 (VERIFY)    → writes diff result
      Station 5 (RECORD)    → reads everything and saves to database

    Each station only writes its own section. Never touches other sections.
    """

    step_index: int
    task_instruction: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Filled during PERCEIVING ───────────────────────────────────────────
    observation: "Observation | None" = None  # what the agent sees
    before_screenshot: "np.ndarray | None" = None  # screenshot before action

    # ── Filled during REASONING ───────────────────────────────────────────
    raw_model_output: str = ""  # raw text from VLM
    action_planned: Action | None = None  # parsed and validated action
    reasoning: str = ""  # extracted from <think> block

    # ── Filled during EXECUTING ───────────────────────────────────────────
    after_screenshot: "np.ndarray | None" = None  # screenshot after action
    execution_success: bool = False  # did OS call succeed?

    # ── Filled during VERIFYING ───────────────────────────────────────────
    diff_result: DiffResult | None = None  # how screen changed

    # ── Filled if something goes wrong ────────────────────────────────────
    error: str | None = None


# ─── Retry Policy ─────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """
    Decides WHEN and HOW LONG to wait before retrying a failed step.

    Uses exponential backoff — each retry waits longer than the last.

    Why exponential backoff?
    If the OS is slow to respond, waiting slightly longer gives it
    more time. If the GPU is busy, waiting longer lets it finish.
    Retrying immediately wastes all retries on the same problem.

    Example with base_delay=0.5 and multiplier=2.0:
      Attempt 1: wait 0.5 seconds
      Attempt 2: wait 1.0 seconds
      Attempt 3: wait 2.0 seconds
      Attempt 4: give up → task FAILED
    """

    max_retries: int = 3  # how many times to retry before giving up permanently
    base_delay_s: float = 0.5  # how long to wait before the first retry (in seconds)
    backoff_multiplier: float = (
        2.0  # how much to multiply the delay for each subsequent retry this create exponential backoff — each retry waits longer than the last
    )
    max_delay_s: float = 8.0  # Never wait more than 8 seconds.

    def delay_for_attempt(self, attempt: int) -> float:
        """
        Calculate how long to wait before the given retry attempt.

        Formula: base * (multiplier ^ (attempt - 1))
          attempt=1: 0.5 * (2.0 ^ 0) = 0.5 * 1  = 0.5s
          attempt=2: 0.5 * (2.0 ^ 1) = 0.5 * 2  = 1.0s
          attempt=3: 0.5 * (2.0 ^ 2) = 0.5 * 4  = 2.0s

        max_delay_s caps the result — we never wait more than 8 seconds.

        Args:
            attempt: 1 for first retry, 2 for second, 3 for third...

        Returns:
            Seconds to wait (float, capped at max_delay_s)
        """
        # Calculate raw delay using exponential formula
        raw_delay = self.base_delay_s * (self.backoff_multiplier ** (attempt - 1))

        # Never wait more than the maximum allowed delay
        return min(raw_delay, self.max_delay_s)

    def should_retry(self, attempt: int, error: str) -> bool:
        """
        Decide whether this failure is worth retrying.

        Two reasons to NOT retry:
          1. We have already retried too many times
          2. The error is permanent — retrying will never fix it

        Permanent errors (safety blocks, parse failures) happen
        because the model made a bad decision — not because of a
        temporary system issue. Retrying will just repeat the same mistake.

        Args:
            attempt: how many times we have already retried
            error:   the error message from the failed step

        Returns:
            True = try again | False = give up permanently
        """
        # Check 1: Have we exhausted all retries?
        if attempt >= self.max_retries:
            log.warning(
                "retry_exhausted",
                attempts=attempt,
                max=self.max_retries,
            )
            return False

        # Check 2: Is this a permanent error that retrying cannot fix?
        error_lower = error.lower()
        permanent_keywords = ["safety", "blocked", "parse", "invalid json"]

        for keyword in permanent_keywords:
            if keyword in error_lower:
                log.warning(
                    "permanent_error_no_retry",
                    error=error,
                    matched_keyword=keyword,
                )
                return False

        # If we get here: attempts remaining AND error is transient → retry
        return True


# ─── Agent Loop ───────────────────────────────────────────────────────────────


class AgentLoop:
    """
    Explicit state machine that runs the agent task execution loop.

    This class ORCHESTRATES — it coordinates all other components.
    It does NOT implement perception, reasoning, or execution itself.

    Think of it like a project manager:
      The manager does not write code, design UI, or test software.
      The manager coordinates developers, designers, and testers.
      AgentLoop coordinates: platform, ocr, model, executor, memory, safety.

    If you change the model → AgentLoop does not change.
    If you change the database → AgentLoop does not change.
    If you change the platform → AgentLoop does not change.
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
        # Store all injected dependencies
        self._platform = platform
        self._ocr = ocr_engine
        self._model = model
        self._executor = executor
        self._memory = memory
        self._safety = safety

        # If no diff_analyzer provided → create one with default settings
        self._diff = diff_analyzer or ScreenDiffAnalyzer()

        self._max_steps = max_steps

        # If no retry policy provided → use defaults (3 retries, 0.5s base)
        self._retry = retry_policy or RetryPolicy()

        # Start in INITIALIZING state — loop has not started yet
        self._state = LoopState.INITIALIZING
        self._retry_count = 0

    @property
    def current_state(self) -> LoopState:
        """Read-only access to current loop state."""
        return self._state

    # ─── State Transition ─────────────────────────────────────────────────────

    def _transition(self, new_state: LoopState, **log_kwargs) -> None:
        """
        Move to a new state and write a structured log entry.

        Every single state change is recorded.
        In production this gives you a complete trace of what happened:

            loop_transition  INITIALIZING → PERCEIVING  step=0
            loop_transition  PERCEIVING   → REASONING   step=0
            loop_transition  REASONING    → EXECUTING   step=0  action=click
            loop_transition  EXECUTING    → VERIFYING   step=0
            loop_transition  VERIFYING    → RECORDING   step=0
            loop_transition  RECORDING    → PERCEIVING  step=1
            ...
            loop_transition  RECORDING    → COMPLETED   steps=5

        When something goes wrong in production, these logs tell you
        exactly where and why it failed.
        """
        old_state = self._state.name
        self._state = new_state

        log.info(
            "loop_transition",
            from_state=old_state,
            to_state=new_state.name,
            **log_kwargs,
        )

    # ─── Main Entry Point ─────────────────────────────────────────────────────

    async def run(
        self,
        task_id: str,
        task_instruction: str,
        history: list[AgentState],
    ) -> TaskResult:
        """
        Run the agent loop from start to finish.

        Keeps running steps until one of these happens:
          - Model says DONE   → return success
          - Model says FAIL   → return failure
          - Max steps reached → return failure
          - Unretriable error → return failure

        Args:
            task_id:          unique ID for this task run
            task_instruction: what the user wants done ("Open Gmail")
            history:          steps from previous sessions (can be empty)

        Returns:
            TaskResult describing what happened
        """
        self._transition(LoopState.INITIALIZING, task_id=task_id)

        # Record start time for elapsed_s calculation
        t0 = time.monotonic()

        # Start with any history from previous sessions
        # We append new steps as we go
        all_steps: list[AgentState] = list(history)

        for step_index in range(self._max_steps):

            # Create fresh context for this step
            ctx = StepContext(
                step_index=step_index,
                task_instruction=task_instruction,
            )

            # Run the full perceive → reason → execute → verify → record cycle
            result = await self._run_step(task_id, ctx, all_steps)

            # ── Case 1: Step completed (may still be success or failure) ──
            if isinstance(result, AgentState):
                all_steps.append(result)
                await self._memory.save_step(task_id, result)

                # Check if this action was DONE or FAIL → terminal state
                terminal = self._check_terminal(result, step_index)
                if terminal is not None:
                    elapsed = time.monotonic() - t0
                    return self._build_result(task_id, terminal, all_steps, elapsed)

                # Step completed normally → reset retry counter
                self._retry_count = 0

            # ── Case 2: Step threw an exception ──────────────────────────
            elif isinstance(result, str):

                should_retry = self._retry.should_retry(self._retry_count, result)

                if should_retry:
                    # Increment retry count and wait before trying again
                    self._retry_count += 1
                    delay = self._retry.delay_for_attempt(self._retry_count)

                    self._transition(
                        LoopState.RETRYING,
                        attempt=self._retry_count,
                        delay_s=delay,
                        error=result,
                    )

                    # Wait for the calculated backoff delay
                    await asyncio.sleep(delay)

                    # step_index -= 1 means: don't count this as a used step
                    # The for loop will increment it back to same value
                    step_index -= 1
                    continue

                else:
                    # No more retries → fail the task permanently
                    elapsed = time.monotonic() - t0
                    return TaskResult(
                        task_id=task_id,
                        success=False,
                        steps=step_index + 1,
                        error=f"Retry exhausted: {result}",
                        elapsed_s=elapsed,
                    )

        # ── Reached max_steps without DONE or FAIL ────────────────────────
        self._transition(LoopState.MAX_STEPS, max_steps=self._max_steps)
        elapsed = time.monotonic() - t0
        return TaskResult(
            task_id=task_id,
            success=False,
            steps=self._max_steps,
            error="Maximum steps reached without completing the task",
            elapsed_s=elapsed,
        )

    # ─── One Complete Step ────────────────────────────────────────────────────

    async def _run_step(
        self,
        task_id: str,
        ctx: StepContext,
        history: list[AgentState],
    ) -> AgentState | str:
        """
        Run one complete step: perceive → reason → execute → verify → record.

        Returns AgentState if the step completed (even if action failed).
        Returns str (error message) if an exception was thrown.

        This method NEVER raises an exception.
        All exceptions are caught and returned as strings.
        The caller (run()) decides what to do with the error.

        Why never raise?
        If this method raised, the entire task would crash.
        We want the loop to handle errors gracefully — not crash.
        """
        try:
            # Each sub-method receives ctx, fills in its section, returns ctx
            self._transition(LoopState.PERCEIVING, step=ctx.step_index)
            ctx = await self._perceive(ctx)

            self._transition(LoopState.REASONING, step=ctx.step_index)
            ctx = await self._reason(ctx, history)

            self._transition(
                LoopState.EXECUTING,
                step=ctx.step_index,
                action=ctx.action_planned.type if ctx.action_planned else None,
            )
            ctx = await self._execute(ctx)

            self._transition(LoopState.VERIFYING, step=ctx.step_index)
            ctx = await self._verify(ctx)

            self._transition(LoopState.RECORDING, step=ctx.step_index)
            state = self._build_state(task_id, ctx)
            return state

        except Exception as exc:
            # Log the full traceback for debugging
            log.error(
                "step_exception",
                step=ctx.step_index,
                error=str(exc),
                exc_info=True,
            )
            # Return error as string — do NOT raise
            return str(exc)

    # ─── PERCEIVE ─────────────────────────────────────────────────────────────

    async def _perceive(self, ctx: StepContext) -> StepContext:
        """
        Capture the screen and build a structured Observation.

        Three data sources are combined:
          1. Screenshot → raw pixel array (what the screen looks like)
          2. OCR text   → list of TextRegion objects (what text is on screen)
          3. UI tree    → list of accessibility elements (what is interactive)

        We also save the screenshot as before_screenshot so that
        the verifier can compare it to the after_screenshot later.
        """
        # Capture screenshot — returns (numpy array, metadata dict)
        screenshot, meta = await self._platform.capture()

        # Save as before_screenshot for later comparison in _verify
        ctx.before_screenshot = screenshot

        # Run OCR to extract text regions from the image
        # OCR is synchronous (CPU-bound) so no await needed
        ocr_regions = self._ocr.run(screenshot)

        # Get accessibility UI tree (buttons, inputs, labels)
        ui_elements = await self._platform.get_ui_tree()

        # Combine all three into one Observation object
        # This is the contract between perception and reasoning
        ctx.observation = fuse(
            image=screenshot,
            meta=meta,
            ocr_regions=ocr_regions,
            ui_elements=ui_elements,
        )

        return ctx

    # ─── REASON ───────────────────────────────────────────────────────────────

    async def _reason(
        self,
        ctx: StepContext,
        history: list[AgentState],
    ) -> StepContext:
        """
        Ask the model what to do next.

        Three sub-steps:
          1. Call model with observation + task + history → raw text
          2. Extract reasoning from <think>...</think> block
          3. Parse the <action>...</action> block into a typed Action
          4. Check safety → block dangerous actions
        """
        # Call the model — returns raw text like:
        # "<think>I see Gmail...</think><action>{"type":"click",...}</action>"
        raw = await self._model.predict_action(
            observation=ctx.observation,
            task=ctx.task_instruction,
            history=[s.__dict__ for s in history[-10:]],  # last 10 steps
            plan=[],
        )
        ctx.raw_model_output = raw

        # ── Extract reasoning from <think> block ──────────────────────────
        # re.DOTALL makes . match newlines too (think block spans multiple lines)
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            ctx.reasoning = think_match.group(1).strip()

        # ── Parse the action ──────────────────────────────────────────────
        try:
            ctx.action_planned = parse_action(raw)

        except ActionParseError as exc:
            # Model output was not parseable → create FAIL action
            # The agent will handle this gracefully
            log.warning(
                "action_parse_failed",
                step=ctx.step_index,
                error=str(exc),
            )
            ctx.action_planned = Action(
                type=ActionType.FAIL,
                description=f"Could not parse model output: {exc}",
            )
            return ctx

        # ── Safety classification ─────────────────────────────────────────
        from visionnav.safety.classifier import RiskLevel

        risk = self._safety.classify(ctx.action_planned)

        if risk == RiskLevel.BLOCKED:
            log.warning(
                "action_blocked_by_safety",
                step=ctx.step_index,
                action_type=ctx.action_planned.type,
            )
            ctx.action_planned = Action(
                type=ActionType.FAIL,
                description=f"Safety blocked action: {ctx.action_planned.type}",
            )

        return ctx

    # ─── EXECUTE ──────────────────────────────────────────────────────────────

    async def _execute(self, ctx: StepContext) -> StepContext:
        """
        Physically perform the planned action on the computer.

        Steps:
          1. Get screen dimensions (needed to convert normalized coords to pixels)
          2. Execute the action via the platform adapter
          3. Capture the after_screenshot for comparison in _verify
        """
        if ctx.action_planned is None:
            ctx.error = "No action planned — cannot execute"
            return ctx

        # Get screen dimensions in pixels
        screen_w, screen_h = self._platform.get_screen_size()

        # Execute the action — returns True if OS call succeeded
        ctx.execution_success = await self._executor.execute(
            ctx.action_planned,
            screen_w,
            screen_h,
        )

        # Capture the screen AFTER the action
        # We compare before vs after in _verify to check if action worked
        after_screenshot, _ = await self._platform.capture()
        ctx.after_screenshot = after_screenshot

        return ctx

    # ─── VERIFY ───────────────────────────────────────────────────────────────

    async def _verify(self, ctx: StepContext) -> StepContext:
        """
        Check if the action had a visible effect on the screen.

        Uses ScreenDiffAnalyzer to classify HOW the screen changed:
          NO_CHANGE    → action had no effect (possible failure)
          MINOR_CHANGE → small change (cursor, clock) — may not be action result
          TEXT_UPDATE  → text changed (typing succeeded)
          NEW_ELEMENT  → new UI element appeared (dialog, button)
          LAYOUT_SHIFT → major layout change (window opened)
          FULL_SCREEN  → entire screen changed (app switched)

        DONE and FAIL actions skip verification — they don't change the screen.
        """
        # Skip verification for actions that don't change the screen
        no_verify_types = {ActionType.DONE, ActionType.FAIL, ActionType.WAIT}
        if ctx.action_planned and ctx.action_planned.type in no_verify_types:
            return ctx

        # Skip if we don't have both screenshots to compare
        if ctx.before_screenshot is None or ctx.after_screenshot is None:
            return ctx

        # Analyze the difference
        ctx.diff_result = self._diff.analyze(
            before=ctx.before_screenshot,
            after=ctx.after_screenshot,
        )

        log.info(
            "screen_verified",
            step=ctx.step_index,
            change_type=ctx.diff_result.change_type.value,
            change_ratio=round(ctx.diff_result.change_ratio, 4),
            is_significant=ctx.diff_result.is_significant,
            summary=ctx.diff_result.summary,
        )

        return ctx

    # ─── BUILD HELPERS ────────────────────────────────────────────────────────

    def _check_terminal(
        self,
        state: AgentState,
        step_index: int,
    ) -> LoopState | None:
        """
        Check if this step reached a terminal condition.

        This is a PURE FUNCTION — no side effects, no async, no logging.
        Pure functions are easy to test and reason about.
        You can test every case without creating a real agent.

        Returns:
            LoopState.COMPLETED  if action was DONE
            LoopState.FAILED     if action was FAIL
            None                 if loop should continue
        """
        if state.action_taken is None:
            return None

        if state.action_taken.type == ActionType.DONE:
            return LoopState.COMPLETED

        if state.action_taken.type == ActionType.FAIL:
            return LoopState.FAILED

        return None

    def _build_state(
        self,
        task_id: str,
        ctx: StepContext,
    ) -> AgentState:
        """
        Convert a completed StepContext into an AgentState for storage.

        AgentState is the permanent record of what happened this step.
        It gets saved to the database and used as history in future steps.

        We extract:
          - What the screen showed (OCR text as a single string)
          - What action was planned and executed
          - Whether the action succeeded
          - Any errors that occurred
          - The exact timestamp
        """
        # Join all OCR region texts into one string for easy storage
        ocr_text = ""
        if ctx.observation and ctx.observation.ocr_regions:
            ocr_text = " | ".join(r.text for r in ctx.observation.ocr_regions)

        return AgentState(
            step_index=ctx.step_index,
            task_instruction=ctx.task_instruction,
            screenshot_path="",  # placeholder — saved separately
            ocr_text=ocr_text,
            action_taken=ctx.action_planned,
            action_success=ctx.execution_success,
            reasoning=ctx.reasoning,
            timestamp=ctx.started_at,
            error=ctx.error,
        )

    def _build_result(
        self,
        task_id: str,
        terminal: LoopState,
        all_steps: list[AgentState],
        elapsed_s: float,
    ) -> TaskResult:
        """
        Build the final TaskResult when a terminal state is reached.

        success = True only if terminal is COMPLETED.
        FAILED and MAX_STEPS both produce success=False.
        """
        success = terminal == LoopState.COMPLETED

        # Log the final transition
        self._transition(
            terminal,
            steps=len(all_steps),
            elapsed_s=round(elapsed_s, 2),
        )

        # Get the description from the last action (e.g. "Opened Gmail successfully")
        last = all_steps[-1] if all_steps else None
        summary = ""
        if last and last.action_taken:
            summary = last.action_taken.description

        return TaskResult(
            task_id=task_id,
            success=success,
            steps=len(all_steps),
            summary=summary,
            error=None if success else "Task did not complete successfully",
            elapsed_s=elapsed_s,
        )
