"""
SQLModel ORM(Object Relational Mapper) models for memory persistence, means
This file contains database models using SQLModel,It converts"Python Objects  <-> Database Tables".
Currently unused in MVP (sqlite.py uses raw aiosqlite).
Ready for Phase 7+ when we switch to SQLModel + PostgreSQL.
why SQLModel?
SQLModel helps Python talk to databases easily.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


# Creates a database table named task with columns:
class Task(SQLModel, table=True):
    task_id: str = Field(primary_key=True)
    instruction: str
    status: str = "running"
    result: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Step(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_id: str = Field(foreign_key="task.task_id")
    step_index: int
    action_type: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# What is aiosqlite?
# A Python library for async SQLite database operations.
# It allows you to perform database queries without blocking the main thread, making it suitable for applications that require concurrency.

# Field
# Used to configure database columns.
# Example:
# Field(primary_key=True)
# means:
# This column is PRIMARY KEY
