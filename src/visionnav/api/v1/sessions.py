"""Session management — stub for Phase 7+."""
from fastapi import APIRouter
import uuid

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("/")
async def create_session() -> dict:
    return {"session_id": str(uuid.uuid4()), "status": "created"}
