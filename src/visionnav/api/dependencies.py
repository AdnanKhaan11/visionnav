"""FastAPI dependency injection."""
from __future__ import annotations
from functools import lru_cache
from fastapi import Header, HTTPException, Depends
from visionnav.settings import Settings


@lru_cache(maxsize=1)
def get_cached_settings() -> Settings:
    return Settings()


async def verify_api_key(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_cached_settings),
) -> str:
    if not settings.api.valid_keys:
        return "dev-open"
    token = authorization.replace("Bearer ", "").strip()
    if token not in settings.api.valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return token


async def get_memory():
    from visionnav.memory.sqlite import SQLiteMemoryStore
    return SQLiteMemoryStore(get_cached_settings().db.url)
