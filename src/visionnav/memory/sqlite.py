"""SQLite memory store — MVP. Replace with postgres.py for production."""
from __future__ import annotations
import json
import aiosqlite
import structlog
from visionnav.agent.state import AgentState

log = structlog.get_logger(__name__)


class SQLiteMemoryStore:
    def __init__(self, db_url: str = "sqlite+aiosqlite:///./visionnav.db") -> None:
        self._path = db_url.replace("sqlite+aiosqlite:///", "")

    async def _conn(self) -> aiosqlite.Connection:
        conn = await aiosqlite.connect(self._path)
        await conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY, instruction TEXT,
            status TEXT DEFAULT 'running', result TEXT,
            created_at TEXT DEFAULT (datetime('now')))""")
        await conn.execute("""CREATE TABLE IF NOT EXISTS steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT, step_index INTEGER, data TEXT,
            created_at TEXT DEFAULT (datetime('now')))""")
        await conn.commit()
        return conn

    async def save_task(self, task_id: str, instruction: str) -> None:
        async with await self._conn() as c:
            await c.execute(
                "INSERT OR IGNORE INTO tasks(task_id,instruction) VALUES(?,?)",
                (task_id, instruction))
            await c.commit()

    async def save_step(self, task_id: str, state: AgentState) -> None:
        data = {
            "step_index": state.step_index, "task_instruction": state.task_instruction,
            "screenshot_path": state.screenshot_path, "ocr_text": state.ocr_text,
            "action_type": state.action_taken.type if state.action_taken else None,
            "action_success": state.action_success, "reasoning": state.reasoning,
            "error": state.error, "timestamp": state.timestamp.isoformat(),
        }
        async with await self._conn() as c:
            await c.execute(
                "INSERT INTO steps(task_id,step_index,data) VALUES(?,?,?)",
                (task_id, state.step_index, json.dumps(data)))
            await c.commit()

    async def get_task_history(self, task_id: str) -> list[AgentState]:
        async with await self._conn() as c:
            async with c.execute(
                "SELECT data FROM steps WHERE task_id=? ORDER BY step_index", (task_id,)
            ) as cur:
                rows = await cur.fetchall()
        return [_to_state(r[0]) for r in rows]

    async def get_recent_steps(self, task_id: str, n: int = 10) -> list[AgentState]:
        return (await self.get_task_history(task_id))[-n:]

    async def mark_task_complete(self, task_id: str, success: bool, result: str) -> None:
        status = "completed" if success else "failed"
        async with await self._conn() as c:
            await c.execute(
                "UPDATE tasks SET status=?,result=? WHERE task_id=?", (status, result, task_id))
            await c.commit()


def _to_state(data_json: str) -> AgentState:
    from datetime import datetime
    d = json.loads(data_json)
    return AgentState(
        step_index=d["step_index"], task_instruction=d["task_instruction"],
        screenshot_path=d["screenshot_path"], ocr_text=d["ocr_text"],
        action_taken=None, action_success=d["action_success"],
        reasoning=d["reasoning"], timestamp=datetime.fromisoformat(d["timestamp"]),
        error=d.get("error"),
    )
