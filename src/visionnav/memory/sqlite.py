"""SQLite memory store — MVP."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite
import structlog

from visionnav.agent.state import AgentState

log = structlog.get_logger(__name__)

# ─── Schema ───────────────────────────────────────────────────────────────────

_CREATE_TASKS = """
    CREATE TABLE IF NOT EXISTS tasks (
        task_id     TEXT PRIMARY KEY,
        instruction TEXT,
        status      TEXT DEFAULT 'running',
        result      TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )
"""

_CREATE_STEPS = """
    CREATE TABLE IF NOT EXISTS steps (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id     TEXT,
        step_index  INTEGER,
        data        TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    )
"""


async def _ensure_schema(conn: aiosqlite.Connection) -> None:
    """Create tables if they don't exist."""
    await conn.execute(_CREATE_TASKS)
    await conn.execute(_CREATE_STEPS)
    await conn.commit()


# ─── Memory Store ─────────────────────────────────────────────────────────────


class SQLiteMemoryStore:
    """
    Persists agent task + step history in a local SQLite database.

    MVP implementation. To switch to PostgreSQL:
    1. Create postgres.py with same interface
    2. Change VISIONNAV_DB__URL in .env
    Zero changes to agent.py or API needed.
    """

    def __init__(self, db_url: str = "sqlite+aiosqlite:///./visionnav.db") -> None:
        self._path = db_url.replace("sqlite+aiosqlite:///", "")

    async def save_task(self, task_id: str, instruction: str) -> None:
        """Create a new task record in the database."""
        async with aiosqlite.connect(self._path) as conn:
            await _ensure_schema(conn)
            await conn.execute(
                "INSERT OR IGNORE INTO tasks(task_id, instruction) VALUES(?, ?)",
                (task_id, instruction),
            )
            await conn.commit()

    async def save_step(self, task_id: str, state: AgentState) -> None:
        """Save one agent step to the database."""
        data = {
            "step_index": state.step_index,
            "task_instruction": state.task_instruction,
            "screenshot_path": state.screenshot_path,
            "ocr_text": state.ocr_text,
            # Save action type as string value (not enum object)
            "action_type": (
                state.action_taken.type.value if state.action_taken else None
            ),
            "action_description": (
                state.action_taken.description if state.action_taken else None
            ),
            "action_success": state.action_success,
            "reasoning": state.reasoning,
            "error": state.error,
            "timestamp": state.timestamp.isoformat(),
        }
        async with aiosqlite.connect(self._path) as conn:
            await _ensure_schema(conn)
            await conn.execute(
                "INSERT INTO steps(task_id, step_index, data) VALUES(?, ?, ?)",
                (task_id, state.step_index, json.dumps(data)),
            )
            await conn.commit()

    async def get_task_history(self, task_id: str) -> list[AgentState]:
        """Get all steps for a task ordered by step index."""
        async with aiosqlite.connect(self._path) as conn:
            await _ensure_schema(conn)
            async with conn.execute(
                "SELECT data FROM steps WHERE task_id=? ORDER BY step_index",
                (task_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [_to_state(r[0]) for r in rows]

    async def get_recent_steps(self, task_id: str, n: int = 10) -> list[AgentState]:
        """Get the last N steps for a task."""
        return (await self.get_task_history(task_id))[-n:]

    async def mark_task_complete(
        self, task_id: str, success: bool, result: str
    ) -> None:
        """Update task status to completed or failed."""
        status = "completed" if success else "failed"
        async with aiosqlite.connect(self._path) as conn:
            await _ensure_schema(conn)
            await conn.execute(
                "UPDATE tasks SET status=?, result=? WHERE task_id=?",
                (status, result, task_id),
            )
            await conn.commit()


# ─── Helper ───────────────────────────────────────────────────────────────────


def _to_state(data_json: str) -> AgentState:
    """Convert raw JSON row back into AgentState object."""
    from visionnav.actions.schema import Action, ActionType

    d = json.loads(data_json)

    # Rebuild Action object from saved string
    action = None
    if d.get("action_type"):
        try:
            action = Action(
                type=ActionType(d["action_type"]),
                description=d.get("action_description", ""),
            )
        except Exception:
            action = None

    return AgentState(
        step_index=d["step_index"],
        task_instruction=d["task_instruction"],
        screenshot_path=d["screenshot_path"],
        ocr_text=d["ocr_text"],
        action_taken=action,
        action_success=d["action_success"],
        reasoning=d["reasoning"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
        error=d.get("error"),
    )
