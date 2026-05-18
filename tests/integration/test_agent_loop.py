"""Integration tests — agent loop."""
from __future__ import annotations
import pytest
from visionnav.agent.agent import VisionNavAgent
from visionnav.agent.state import TaskResult
from visionnav.safety.classifier import SafetyClassifier
from visionnav.settings import AgentSettings


def _make(mock_model, mock_platform, tmp_path, mocker):
    from visionnav.memory.sqlite import SQLiteMemoryStore
    return VisionNavAgent(
        model=mock_model, platform=mock_platform,
        memory=SQLiteMemoryStore(f"sqlite+aiosqlite:///{tmp_path}/t.db"),
        safety=SafetyClassifier(),
        settings=AgentSettings(max_steps=10,
                               screenshot_dir=str(tmp_path / "ss")),
    )

@pytest.mark.integration
async def test_done_on_first_step(mock_model, mock_platform, tmp_path, mocker):
    agent  = _make(mock_model, mock_platform, tmp_path, mocker)
    result = await agent.run("t1", "Open calculator")
    assert result.success is True and result.steps == 1

@pytest.mark.integration
async def test_fail_action(mock_model, mock_platform, tmp_path, mocker):
    mock_model.predict_action.return_value = (
        "<think>nope</think>"
        '<action>{"type":"fail","description":"impossible"}</action>'
    )
    result = await _make(mock_model, mock_platform, tmp_path, mocker).run("t2", "x")
    assert result.success is False

@pytest.mark.integration
async def test_max_steps(mock_model, mock_platform, tmp_path, mocker):
    mock_model.predict_action.return_value = (
        "<think>clicking</think>"
        '<action>{"type":"click","coordinates":[0.5,0.5]}</action>'
    )
    result = await _make(mock_model, mock_platform, tmp_path, mocker).run("t3", "forever")
    assert result.steps == 10 and result.success is False
