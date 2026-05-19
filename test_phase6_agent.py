"""
Phase 6 — Full Agent Loop Test
Tests the complete agent running a real task end to end.
"""

import sys
import asyncio

sys.path.insert(0, "src")

from visionnav.agent.agent import VisionNavAgent
from visionnav.memory.sqlite import SQLiteMemoryStore
from visionnav.platforms.desktop import DesktopPlatform
from visionnav.safety.classifier import SafetyClassifier
from visionnav.settings import AgentSettings, ModelSettings
from visionnav.models.local import LocalModelBackend


async def main():
    print("=" * 55)
    print("VisionNav — Phase 6 Full Agent Loop Test")
    print("=" * 55)

    # Build agent with all real components
    print("\nLoading components...")

    model = LocalModelBackend(
        ModelSettings(
            name="microsoft/DialoGPT-small",
            dtype="float32",
            device_map="cpu",
        )
    )

    platform = DesktopPlatform()
    memory = SQLiteMemoryStore("sqlite+aiosqlite:///./phase6_test.db")
    safety = SafetyClassifier()
    settings = AgentSettings(
        max_steps=5,
        screenshot_dir="data/phase6_screenshots",
    )

    agent = VisionNavAgent(
        model=model,
        platform=platform,
        memory=memory,
        safety=safety,
        settings=settings,
    )

    print("All components loaded!")
    print("\nStarting task in 3 seconds...")
    print("Watch your screen!")
    import time

    time.sleep(3)

    # Run the task
    import uuid

    task_id = str(uuid.uuid4())
    task = "Open Notepad"

    print(f"\nTask ID : {task_id[:8]}...")
    print(f"Task    : {task}")
    print(f"Max steps: {settings.max_steps}")
    print("-" * 55)

    result = await agent.run(task_id, task)

    # Show result
    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"Success  : {result.success}")
    print(f"Steps    : {result.steps}")
    print(f"Summary  : {result.summary or result.error}")

    # Show step history
    print("\nStep History:")
    history = await memory.get_task_history(task_id)
    for s in history:
        action = s.action_taken.type if s.action_taken else "unknown"
        status = "✅" if s.action_success else "❌"
        print(f"  [{s.step_index}] {action} {status}")
        if s.error:
            print(f"       Error: {s.error}")

    print("\nPhase 6 test complete!")


asyncio.run(main())
