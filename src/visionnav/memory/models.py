"""
SQLModel ORM models for memory persistence.
Currently unused in MVP (sqlite.py uses raw aiosqlite).
Ready for Phase 7+ when we switch to SQLModel + PostgreSQL.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Task(SQLModel, table=True):
    task_id:     str           = Field(primary_key=True)
    instruction: str
    status:      str           = "running"
    result:      Optional[str] = None
    created_at:  datetime      = Field(default_factory=datetime.utcnow)


class Step(SQLModel, table=True):
    id:          Optional[int] = Field(default=None, primary_key=True)
    task_id:     str           = Field(foreign_key="task.task_id")
    step_index:  int
    action_type: Optional[str] = None
    success:     bool          = False
    error:       Optional[str] = None
    created_at:  datetime      = Field(default_factory=datetime.utcnow)
