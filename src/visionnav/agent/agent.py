"""VisionNavAgent — main agent loop (perceive → reason → act → verify)."""

from __future__ import annotations
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from visionnav.actions.executor import ActionExecutor
from visionnav.actions.parser import (
    ActionParseError,
    parse_action,
)
from visionnav.actions.schema import Action, ActionType
from visionnav.actions.verifier import ActionVerifier
from visionnav.agent.planner import TaskPlanner
from visionnav.agent.reporter import TaskReporter
from visionnav.agent.state import AgentState, TaskResult
from visionnav.memory.base import MemoryStore
from visionnav.models.base import ModelBackend
from visionnav.perception.fusion import fuse
from visionnav.perception.ocr import OCREngine
from visionnav.platforms.base import PlatformAdapter
from visionnav.safety.classifier import RiskLevel, SafetyClassifier
from visionnav.settings import AgentSettings
from visionnav.utils.image import save_screenshot
from visionnav.utils.logging import get_logger

log = get_logger(__name__)


def _extract_reasoning(text: str) -> str:
    import re

    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


class VisionNavAgent:
    """
    Core agent. Inject all dependencies — nothing is constructed internally.
    Swap model/platform/memory implementations without touching this file.
    """

    def __init__(
        self,
        model: ModelBackend,
        platform: PlatformAdapter,
        memory: MemoryStore,
        safety: SafetyClassifier,
        settings: AgentSettings,
    ) -> None:
        self._model = model
        self._platform = platform
        self._memory = memory
        self._safety = safety
        self._settings = settings
        self._ocr = OCREngine()
        self._executor = ActionExecutor(platform)
        self._verifier = ActionVerifier(settings.change_threshold)
        self._planner = TaskPlanner()
        self._reporter = TaskReporter()

    async def run(self, task_id: str, task: str) -> TaskResult:
        bound = log.bind(task_id=task_id)
        bound.info("task_started", instruction=task[:100])

        await self._memory.save_task(task_id, task)
        plan = self._planner.decompose(task)
        history: list[AgentState] = []

        ss_dir = Path(self._settings.screenshot_dir) / task_id
        ss_dir.mkdir(parents=True, exist_ok=True)

        for step_num in range(self._settings.max_steps):
            bound.info("step_started", step=step_num)

            t0 = datetime.now(timezone.utc)

            # 1. PERCEIVE
            arr, meta = await self._platform.capture()
            ss_path = ss_dir / f"step_{step_num:03d}.png"
            save_screenshot(arr, ss_path)
            meta["path"] = str(ss_path)
            ocr_regions = self._ocr.run(arr)
            ui_elements = await self._platform.get_ui_tree()
            observation = fuse(arr, meta, ocr_regions, ui_elements)

            # 2. REASON
            history_dicts = [s.to_history_entry() for s in history[-10:]]
            raw_output = await self._model.predict_action(
                observation, task, history_dicts, plan
            )
            reasoning = _extract_reasoning(raw_output)

            # 3. PARSE
            try:
                action = parse_action(raw_output)
            except ActionParseError as exc:
                bound.warning("parse_failed", step=step_num, error=str(exc))
                action = Action(
                    type=ActionType.FAIL,
                    description=f"Model output unparseable: {exc}",
                )

            bound.info("action_planned", step=step_num, action=action.type)

            # 4. SAFETY
            risk = self._safety.classify(action, context=observation.to_text_summary())
            if risk >= RiskLevel.HIGH:
                bound.warning("action_blocked", action=action.type, risk=risk.name)
                action = Action(
                    type=ActionType.FAIL,
                    description=f"High-risk action blocked ({risk.name}): {action.description}",
                )

            # 5. EXECUTE
            w, h = self._platform.get_screen_size()
            await self._executor.execute(action, w, h)

            # 6. VERIFY
            after_arr, _ = await self._platform.capture()
            success, change = self._verifier.verify(arr, after_arr, action)

            elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
            bound.info(
                "step_complete",
                step=step_num,
                action=action.type,
                success=success,
                change=round(change, 3),
                elapsed_s=round(elapsed, 2),
            )

            # 7. RECORD
            state = AgentState(
                step_index=step_num,
                task_instruction=task,
                screenshot_path=str(ss_path),
                ocr_text=observation.to_text_summary()[:500],
                action_taken=action,
                action_success=success,
                reasoning=reasoning[:1000],
                error=None if success else f"No effect detected (change={change:.3f})",
            )
            await self._memory.save_step(task_id, state)
            history.append(state)

            # 8. TERMINAL
            if action.type == ActionType.DONE:
                await self._memory.mark_task_complete(task_id, True, action.description)
                bound.info("task_completed", steps=step_num + 1)
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    steps=step_num + 1,
                    summary=action.description,
                )

            if action.type == ActionType.FAIL:
                await self._memory.mark_task_complete(
                    task_id, False, action.description
                )
                bound.warning(
                    "task_failed", steps=step_num + 1, reason=action.description
                )
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    steps=step_num + 1,
                    error=action.description,
                )

        await self._memory.mark_task_complete(task_id, False, "Max steps reached")
        bound.warning("task_max_steps", max=self._settings.max_steps)
        return TaskResult(
            task_id=task_id,
            success=False,
            steps=self._settings.max_steps,
            error="Maximum steps reached without completing the task.",
        )

    async def run_with_loop(
        self,
        task_id: str,
        instruction: str,
    ) -> TaskResult:
        """
        Run the task using the new explicit AgentLoop state machine.

        This is the modern replacement for run().
        The old run() uses an implicit if/elif state machine.
        This uses an explicit, logged, retrying AgentLoop.

        Both methods produce the same TaskResult.
        Use this for new features. Keep run() for existing tests.
        """
        from visionnav.agent.loop import AgentLoop, RetryPolicy

        # Build the loop — inject all dependencies
        loop = AgentLoop(
            platform=self._platform,
            ocr_engine=self._ocr,
            model=self._model,
            executor=self._executor,
            memory=self._memory,
            safety=self._safety,
            max_steps=self._settings.max_steps,
            retry_policy=RetryPolicy(
                max_retries=3,
                base_delay_s=0.5,
            ),
        )

        # Save task to database before starting
        # This creates the task record that GET /v1/tasks/{id} will find
        await self._memory.save_task(task_id, instruction)

        # Load previous steps if this task was already started
        # Allows resuming a task from where it left off
        history = await self._memory.get_recent_steps(task_id, n=10)

        # Run the loop — returns when DONE, FAIL, or MAX_STEPS
        return await loop.run(task_id, instruction, history)


async def run_cli() -> None:
    """Entry point: visionnav-agent 'Open Chrome'"""
    import sys
    from visionnav.settings import Settings
    from visionnav.models.local import LocalModelBackend
    from visionnav.platforms.desktop import DesktopPlatform
    from visionnav.memory.sqlite import SQLiteMemoryStore
    from visionnav.safety.classifier import SafetyClassifier

    if len(sys.argv) < 2:
        print("Usage: visionnav-agent 'Your task here'")
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    settings = Settings()
    agent = VisionNavAgent(
        model=LocalModelBackend(settings.model),
        platform=DesktopPlatform(),
        memory=SQLiteMemoryStore(settings.db.url),
        safety=SafetyClassifier(),
        settings=settings.agent,
    )
    result = await agent.run(str(uuid.uuid4()), task)
    print(f"\n{'✅ Success' if result.success else '❌ Failed'}")
    print(f"Steps: {result.steps}")
    if result.error:
        print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(run_cli())
