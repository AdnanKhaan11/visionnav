"""Unit tests — safety classifier."""
from visionnav.actions.schema import Action, ActionType
from visionnav.safety.classifier import RiskLevel, SafetyClassifier


def test_scroll_safe():
    assert SafetyClassifier().classify(
        Action(type=ActionType.SCROLL, direction="down")) == RiskLevel.SAFE

def test_click_low():
    assert SafetyClassifier().classify(
        Action(type=ActionType.CLICK, coordinates=(0.5,0.5))) == RiskLevel.LOW

def test_delete_click_high():
    assert SafetyClassifier().classify(
        Action(type=ActionType.CLICK, coordinates=(0.5,0.5), description="click delete")) == RiskLevel.HIGH

def test_password_type_high():
    assert SafetyClassifier().classify(
        Action(type=ActionType.TYPE, text="x"), context="password field") == RiskLevel.HIGH

def test_done_safe():
    assert SafetyClassifier().classify(
        Action(type=ActionType.DONE)) == RiskLevel.SAFE
