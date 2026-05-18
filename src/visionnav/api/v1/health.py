"""Health check endpoints."""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

@router.get("/health",  response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@router.get("/ready",   response_model=HealthResponse)
async def ready() -> HealthResponse:
    return HealthResponse(status="ok")
