"""FastAPI dependency injection."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header, HTTPException

from visionnav.settings import Settings


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    """Cached settings for production use."""
    return Settings()


def get_settings() -> Settings:
    """
    Non-cached settings — used in tests so test settings are not ignored.
    In production get_cached_settings is faster.
    In tests each call returns fresh settings with test values.
    """
    return Settings()


async def verify_api_key(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate API key from Authorization header.

    MVP: Static key validation.
    Future: Replace with JWT verification here only.
    Zero changes to any router needed.

    Returns "dev-open" if no keys configured (development mode).
    """
    if not settings.api.valid_keys:
        return "dev-open"

    token = authorization.replace("Bearer ", "").strip()

    if token not in settings.api.valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )
    return token


async def get_memory():
    """Provide MemoryStore instance. Swap SQLite for PostgreSQL here only."""
    from visionnav.memory.sqlite import SQLiteMemoryStore

    return SQLiteMemoryStore(get_cached_settings().db.url)
