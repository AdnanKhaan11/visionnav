"""Aggregate all v1 routers."""
from fastapi import APIRouter
from visionnav.api.v1 import health, sessions, tasks

v1_router = APIRouter()
v1_router.include_router(health.router)
v1_router.include_router(tasks.router)
v1_router.include_router(sessions.router)
