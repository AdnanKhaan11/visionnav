"""ModelBackend Protocol — the interface every VLM backend must satisfy.
Every Vision Language Model (VLM) backend must follow this interface.
like claude,chatgpt,gemini, etc.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from visionnav.perception.fusion import Observation


@runtime_checkable
class ModelBackend(Protocol):
    async def predict_action(
        self,
        observation: Observation,
        task: str,
        history: list[dict],
        plan: list[str],
    ) -> str: ...
