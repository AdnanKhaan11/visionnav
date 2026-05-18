"""Safety gate helpers."""
from __future__ import annotations
from visionnav.safety.classifier import RiskLevel


def should_block(risk: RiskLevel) -> bool:
    return risk >= RiskLevel.BLOCKED

def should_confirm(risk: RiskLevel, confirmation_required: bool = True) -> bool:
    return confirmation_required and risk >= RiskLevel.HIGH
