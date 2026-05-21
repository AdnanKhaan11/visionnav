"""MemoryStore Protocol — swap SQLite → PostgreSQL by changing one file."""

from __future__ import annotations
from typing import Protocol
from visionnav.agent.state import AgentState


class MemoryStore(Protocol):
    async def save_task(self, task_id: str, instruction: str) -> None: ...
    async def save_step(self, task_id: str, state: AgentState) -> None: ...
    async def get_task_history(self, task_id: str) -> list[AgentState]: ...
    async def get_recent_steps(self, task_id: str, n: int = 10) -> list[AgentState]: ...
    async def mark_task_complete(
        self, task_id: str, success: bool, result: str
    ) -> None: ...


# What is Protocol?
# Protocol defines:
# "What methods a class MUST have"
# NOT actual implementation.
# Only rules/contract.
