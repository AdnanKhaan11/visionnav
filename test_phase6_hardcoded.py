"""
Test agent execution with hardcoded actions.
Proves infrastructure works — model intelligence comes after fine-tuning.
"""

import sys
import asyncio

sys.path.insert(0, "src")

from visionnav.agent.agent import VisionNavAgent
from visionnav.memory.sqlite import SQLiteMemoryStore
from visionnav.platforms.desktop import DesktopPlatform
from visionnav.safety.classifier import SafetyClassifier
from visionnav.settings import AgentSettings
from visionnav.perception.fusion import Observation
import uuid
import numpy as np
import base64, io
from PIL import Image


class HardcodedModel:
    """
    Fake model that returns correct actions in sequence.
    Used to test agent infrastructure without fine-tuned model.
    This simulates what a trained model would output.
    """

    def __init__(self, actions: list[str]):
        self._actions = actions
        self._index = 0

    async def predict_action(self, observation, task, history, plan) -> str:
        if self._index >= len(self._actions):
            return '<action>{"type":"done","description":"All steps complete"}</action>'

        action = self._actions[self._index]
        self._index += 1
        return action


async def main():
    print("=" * 55)
    print("Phase 6 — Hardcoded Actions Test")
    print("Proves agent infrastructure works correctly")
    print("=" * 55)

    # Define exactly what a trained model would output
    # for the task "Open Notepad"
    trained_model_outputs = [
        "<think>I need to open Run dialog first.</think>\n"
        '<action>{"type":"key","key":"win+r","description":"Open Run dialog"}</action>',
        "<think>Run dialog is open. Type notepad.</think>\n"
        '<action>{"type":"type","text":"notepad","description":"Type notepad"}</action>',
        "<think>notepad is typed. Press Enter to open.</think>\n"
        '<action>{"type":"key","key":"enter","description":"Press Enter"}</action>',
        "<think>Notepad should be open now. Task complete.</think>\n"
        '<action>{"type":"done","description":"Notepad opened successfully"}</action>',
    ]

    model = HardcodedModel(trained_model_outputs)
    platform = DesktopPlatform()
    memory = SQLiteMemoryStore("sqlite+aiosqlite:///./phase6_hardcoded.db")
    safety = SafetyClassifier()
    settings = AgentSettings(
        max_steps=10,
        screenshot_dir="data/phase6_screenshots",
    )

    agent = VisionNavAgent(
        model=model,
        platform=platform,
        memory=memory,
        safety=safety,
        settings=settings,
    )

    print("\nStarting in 3 seconds...")
    print("Watch your screen — Notepad will open!")
    import time

    time.sleep(3)

    task_id = str(uuid.uuid4())
    result = await agent.run(task_id, "Open Notepad")

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"Success : {result.success}")
    print(f"Steps   : {result.steps}")
    print(f"Summary : {result.summary}")

    print("\nStep History:")
    history = await memory.get_task_history(task_id)
    for s in history:
        action = s.action_taken.type if s.action_taken else "none"
        status = "✅" if s.action_success else "❌"
        print(f"  [{s.step_index}] {action} {status}")


asyncio.run(main())
