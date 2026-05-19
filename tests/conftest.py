"""Shared pytest fixtures."""

from __future__ import annotations
import numpy as np
import pytest
from visionnav.actions.schema import Action, ActionType


@pytest.fixture
def blank_screen() -> np.ndarray:
    return np.zeros((600, 800, 3), dtype=np.uint8)


@pytest.fixture
def changed_screen() -> np.ndarray:
    return np.full((600, 800, 3), 255, dtype=np.uint8)


@pytest.fixture
def test_observation(blank_screen):
    import base64, io
    from PIL import Image
    from visionnav.perception.fusion import Observation

    buf = io.BytesIO()
    Image.fromarray(blank_screen).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return Observation(
        screenshot_b64=b64,
        screenshot_path="/tmp/test.png",
        screen_width=800,
        screen_height=600,
        platform="linux",
    )


@pytest.fixture
def click_action() -> Action:
    return Action(type=ActionType.CLICK, coordinates=(0.5, 0.5), description="test")


@pytest.fixture
def done_action() -> Action:
    return Action(type=ActionType.DONE, description="Task complete")


@pytest.fixture
def mock_model(mocker):
    m = mocker.AsyncMock()
    m.predict_action.return_value = (
        "<think>Done.</think>" '<action>{"type":"done","description":"Done"}</action>'
    )
    return m


@pytest.fixture
def mock_platform(mocker, blank_screen):
    p = mocker.AsyncMock()
    p.capture.return_value = (blank_screen, {"width": 800, "height": 600})
    p.get_ui_tree.return_value = []
    p.get_screen_size = lambda: (800, 600)
    p.execute_click.return_value = True
    p.execute_type.return_value = True
    p.execute_scroll.return_value = True
    p.execute_key.return_value = True
    return p
