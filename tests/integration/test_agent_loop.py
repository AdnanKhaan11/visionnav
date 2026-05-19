"""Integration tests — agent loop."""

from __future__ import annotations
import pytest
from pathlib import Path
from visionnav.agent.agent import VisionNavAgent
from visionnav.safety.classifier import SafetyClassifier
from visionnav.settings import AgentSettings

# Use project folder instead of Windows temp (avoids PermissionError)
TEST_DIR = Path("D:/visionnav/tests/tmp")
TEST_DIR.mkdir(parents=True, exist_ok=True)


def _make(mock_model, mock_platform, test_id: str):
    from visionnav.memory.sqlite import SQLiteMemoryStore

    db_path = TEST_DIR / f"{test_id}.db"
    return VisionNavAgent(
        model=mock_model,
        platform=mock_platform,
        memory=SQLiteMemoryStore(f"sqlite+aiosqlite:///{db_path}"),
        safety=SafetyClassifier(),
        settings=AgentSettings(
            max_steps=10,
            screenshot_dir=str(TEST_DIR / "screenshots"),
        ),
    )


@pytest.mark.integration
async def test_done_on_first_step(mock_model, mock_platform):
    """Agent completes successfully when model returns DONE."""
    agent = _make(mock_model, mock_platform, "t1")
    result = await agent.run("t1", "Open calculator")
    assert result.success is True
    assert result.steps == 1


@pytest.mark.integration
async def test_fail_action(mock_model, mock_platform):
    """Agent fails gracefully when model returns FAIL."""
    mock_model.predict_action.return_value = (
        "<think>nope</think>"
        '<action>{"type":"fail","description":"impossible"}</action>'
    )
    result = await _make(mock_model, mock_platform, "t2").run("t2", "x")
    assert result.success is False


@pytest.mark.integration
async def test_max_steps(mock_model, mock_platform):
    """Agent stops after max_steps without DONE or FAIL."""
    mock_model.predict_action.return_value = (
        "<think>clicking</think>"
        '<action>{"type":"click","coordinates":[0.5,0.5]}</action>'
    )
    result = await _make(mock_model, mock_platform, "t3").run("t3", "forever")
    assert result.steps == 10
    assert result.success is False


@pytest.mark.integration
async def test_agent_saves_steps_to_db(mock_model, mock_platform):
    """Every step must be saved to database."""
    from visionnav.memory.sqlite import SQLiteMemoryStore

    db_path = TEST_DIR / "save_test.db"
    memory = SQLiteMemoryStore(f"sqlite+aiosqlite:///{db_path}")

    agent = VisionNavAgent(
        model=mock_model,
        platform=mock_platform,
        memory=memory,
        safety=SafetyClassifier(),
        settings=AgentSettings(
            max_steps=5,
            screenshot_dir=str(TEST_DIR / "ss"),
        ),
    )

    task_id = "save-001"
    await agent.run(task_id, "Open Chrome")

    steps = await memory.get_task_history(task_id)
    assert len(steps) >= 1
    assert steps[0].task_instruction == "Open Chrome"


@pytest.mark.integration
async def test_agent_handles_invalid_model_output(mock_model, mock_platform):
    """Agent must not crash on garbage model output."""
    mock_model.predict_action.return_value = (
        "this is completely invalid output with no action block"
    )
    agent = _make(mock_model, mock_platform, "t4")
    result = await agent.run("t4", "Do something")
    assert result is not None
    assert result.success is False
    assert result.steps >= 1


@pytest.mark.integration
async def test_agent_respects_task_instruction(mock_model, mock_platform):
    """Task instruction must be saved and retrievable."""
    from visionnav.memory.sqlite import SQLiteMemoryStore

    db_path = TEST_DIR / "instr_test.db"
    memory = SQLiteMemoryStore(f"sqlite+aiosqlite:///{db_path}")

    agent = VisionNavAgent(
        model=mock_model,
        platform=mock_platform,
        memory=memory,
        safety=SafetyClassifier(),
        settings=AgentSettings(
            max_steps=5,
            screenshot_dir=str(TEST_DIR / "ss"),
        ),
    )

    task = "Open Notepad and type Hello World"
    task_id = "instr-001"
    await agent.run(task_id, task)

    steps = await memory.get_task_history(task_id)
    assert len(steps) >= 1
    assert steps[0].task_instruction == task
