import json
import numpy as np
import pytest
from pathlib import Path
from visionnav.actions.recorder import ActionRecorder, RecordedStep
from visionnav.actions.schema import Action, ActionType


def make_screenshot() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def make_action(action_type: ActionType = ActionType.CLICK) -> Action:
    return Action(
        type=action_type,
        coordinates=(0.5, 0.5) if action_type == ActionType.CLICK else None,
        description="test action",
    )


# Test 1: Recorder saves steps to JSONL
def test_recorder_saves_steps(tmp_path):
    recorder = ActionRecorder(
        task="Open Notepad",
        output_dir=tmp_path,
    )
    recorder.start()
    recorder.record_step(
        make_action(ActionType.KEY),
        make_screenshot(),
        "desktop text",
        {"width": 100, "height": 100},
    )
    recorder.record_step(
        make_action(ActionType.DONE),
        make_screenshot(),
        "notepad open",
        {"width": 100, "height": 100},
    )
    recorder.finish(success=True)

    assert recorder.output_path.exists()
    lines = recorder.output_path.read_text().strip().split("\n")
    assert len(lines) == 2  # two steps = two lines


# Test 2: Each line is valid JSON
def test_recorder_writes_valid_json(tmp_path):
    recorder = ActionRecorder(task="Test task", output_dir=tmp_path)
    recorder.start()
    recorder.record_step(
        make_action(), make_screenshot(), "text", {"width": 100, "height": 100}
    )
    recorder.finish()

    lines = recorder.output_path.read_text().strip().split("\n")
    for line in lines:
        data = json.loads(line)  # must not raise
        assert "step_index" in data
        assert "action" in data
        assert "task" in data


# Test 3: Screenshots are saved to disk
def test_recorder_saves_screenshots(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    recorder.start()
    recorder.record_step(
        make_action(), make_screenshot(), "", {"width": 100, "height": 100}
    )
    recorder.finish()

    screenshots = list(tmp_path.glob("screenshots/**/*.png"))
    assert len(screenshots) == 1


# Test 4: Context manager works correctly
def test_recorder_context_manager(tmp_path):
    with ActionRecorder(task="Test", output_dir=tmp_path) as recorder:
        recorder.record_step(
            make_action(), make_screenshot(), "", {"width": 100, "height": 100}
        )
    assert recorder.output_path.exists()


# Test 5: record_step without start raises
def test_record_without_start_raises(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    with pytest.raises(RuntimeError):
        recorder.record_step(
            make_action(), make_screenshot(), "", {"width": 100, "height": 100}
        )


# Test 6: load() returns steps in order
def test_load_returns_steps_in_order(tmp_path):
    recorder = ActionRecorder(task="Test", output_dir=tmp_path)
    recorder.start()
    for i in range(5):
        recorder.record_step(
            make_action(), make_screenshot(), f"text {i}", {"width": 100, "height": 100}
        )
    recorder.finish()

    loaded = list(recorder.load(recorder.output_path))
    assert len(loaded) == 5
    assert [s.step_index for s in loaded] == [0, 1, 2, 3, 4]
