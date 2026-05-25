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
from visionnav.agent.loop import AgentLoop, LoopState, RetryPolicy, StepContext
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
            platform=MagicMock(),
            ocr_engine=MagicMock(),
            model=MagicMock(),
            executor=MagicMock(),
            memory=MagicMock(),
            safety=MagicMock(),
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
        loop = self._make_loop()
        state = self._make_state(ActionType.DONE)
        assert loop._check_terminal(state, 0) == LoopState.COMPLETED

    def test_fail_action_returns_failed(self):
        loop = self._make_loop()
        state = self._make_state(ActionType.FAIL)
        assert loop._check_terminal(state, 0) == LoopState.FAILED

    def test_click_action_returns_none(self):
        loop = self._make_loop()
        state = self._make_state(ActionType.CLICK)
        assert loop._check_terminal(state, 0) is None

    def test_no_action_returns_none(self):
        from datetime import datetime, timezone

        loop = self._make_loop()
        state = AgentState(
            step_index=0,
            task_instruction="test",
            screenshot_path="",
            ocr_text="",
            action_taken=None,
            action_success=False,
            reasoning="",
            timestamp=datetime.now(timezone.utc),
            error=None,
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
        platform.get_ui_tree.return_value = []
        platform.get_screen_size = lambda: (100, 100)
        platform.execute_click.return_value = True
        platform.execute_type.return_value = True
        platform.execute_key.return_value = True
        return platform

    def _make_mock_model(self, output: str):
        model = AsyncMock()
        model.predict_action.return_value = output
        return model

    def _make_mock_memory(self):
        memory = AsyncMock()
        memory.get_recent_steps.return_value = []
        memory.save_task.return_value = None
        memory.save_step.return_value = None
        memory.mark_task_complete.return_value = None
        return memory

    @pytest.mark.asyncio
    async def test_done_on_first_step_succeeds(self):
        """Loop completes when model returns DONE immediately."""
        loop = AgentLoop(
            platform=self._make_mock_platform(),
            ocr_engine=MagicMock(run=lambda x: []),
            model=self._make_mock_model(
                "<think>Task done</think>"
                '<action>{"type":"done","description":"Completed"}</action>'
            ),
            executor=AsyncMock(execute=AsyncMock(return_value=True)),
            memory=self._make_mock_memory(),
            safety=MagicMock(classify=MagicMock(return_value=0)),
            max_steps=5,
        )
        result = await loop.run("task-1", "Do something", [])
        assert result.success is True
        assert result.steps == 1

    @pytest.mark.asyncio
    async def test_fail_action_terminates_loop(self):
        """Loop terminates with failure when model returns FAIL."""
        loop = AgentLoop(
            platform=self._make_mock_platform(),
            ocr_engine=MagicMock(run=lambda x: []),
            model=self._make_mock_model(
                '<action>{"type":"fail","description":"Cannot do this"}</action>'
            ),
            executor=AsyncMock(execute=AsyncMock(return_value=True)),
            memory=self._make_mock_memory(),
            safety=MagicMock(classify=MagicMock(return_value=0)),
            max_steps=5,
        )
        result = await loop.run("task-2", "Impossible task", [])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_max_steps_terminates_loop(self):
        """Loop stops after max_steps if never completed."""
        loop = AgentLoop(
            platform=self._make_mock_platform(),
            ocr_engine=MagicMock(run=lambda x: []),
            model=self._make_mock_model(
                '<action>{"type":"click","coordinates":[0.5,0.5]}</action>'
            ),
            executor=AsyncMock(execute=AsyncMock(return_value=True)),
            memory=self._make_mock_memory(),
            safety=MagicMock(classify=MagicMock(return_value=0)),
            max_steps=3,
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
            platform=self._make_mock_platform(),
            ocr_engine=MagicMock(run=lambda x: []),
            model=model,
            executor=AsyncMock(execute=AsyncMock(return_value=True)),
            memory=self._make_mock_memory(),
            safety=MagicMock(classify=MagicMock(return_value=0)),
            max_steps=10,
            retry_policy=RetryPolicy(
                max_retries=3,
                base_delay_s=0.01,  # fast in tests
            ),
        )
        result = await loop.run("task-4", "Retry test", [])
        assert result.success is True
        assert call_count["n"] == 2  # called twice: once failed, once succeeded

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
