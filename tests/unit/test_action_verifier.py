"""Unit tests — action verifier."""
import numpy as np
import pytest
from visionnav.actions.schema import Action, ActionType
from visionnav.actions.verifier import ActionVerifier


@pytest.fixture
def verifier():
    return ActionVerifier(change_threshold=0.01)

def test_detects_change(verifier):
    before = np.zeros((100,100,3), dtype=np.uint8)
    after  = np.full((100,100,3), 255, dtype=np.uint8)
    ok, r  = verifier.verify(before, after, Action(type=ActionType.CLICK, coordinates=(0.5,0.5)))
    assert ok is True and r > 0.5

def test_detects_no_change(verifier):
    s     = np.zeros((100,100,3), dtype=np.uint8)
    ok, r = verifier.verify(s, s.copy(), Action(type=ActionType.CLICK, coordinates=(0.5,0.5)))
    assert ok is False and r == 0.0

def test_done_always_ok(verifier):
    s     = np.zeros((100,100,3), dtype=np.uint8)
    ok, _ = verifier.verify(s, s.copy(), Action(type=ActionType.DONE))
    assert ok is True
