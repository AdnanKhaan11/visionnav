"""Unit tests — Action schema validation."""
import pytest
from pydantic import ValidationError
from visionnav.actions.schema import Action, ActionType


def test_valid_click():
    a = Action(type=ActionType.CLICK, coordinates=(0.5, 0.5))
    assert a.type == ActionType.CLICK

def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Action(type=ActionType.CLICK, confidence=1.5)
    with pytest.raises(ValidationError):
        Action(type=ActionType.CLICK, confidence=-0.1)

def test_default_confidence():
    a = Action(type=ActionType.DONE)
    assert a.confidence == 1.0

def test_all_action_types_valid():
    for at in ActionType:
        a = Action(type=at)
        assert a.type == at
