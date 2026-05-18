"""Task management endpoints."""
from __future__ import annotations
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from visionnav.api.dependencies import get_memory, verify_api_key
from visionnav.settings import Settings

router = APIRouter(prefix="/tasks", tags=["tasks"])


class TaskRequest(BaseModel):
    instruction: str
    platform:    str = "desktop"
    max_steps:   int = 50


class TaskResponse(BaseModel):
    task_id: str
    status:  str
    message: str


@router.post("/", response_model=TaskResponse, status_code=202)
async def submit_task(
    body: TaskRequest,
    background_tasks: BackgroundTasks,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> TaskResponse:
    task_id = str(uuid.uuid4())

    async def run_task() -> None:
        from visionnav.agent.agent import VisionNavAgent
        from visionnav.memory.sqlite import SQLiteMemoryStore
        from visionnav.models.local import LocalModelBackend
        from visionnav.platforms.desktop import DesktopPlatform
        from visionnav.safety.classifier import SafetyClassifier
        s     = Settings()
        agent = VisionNavAgent(
            model=LocalModelBackend(s.model), platform=DesktopPlatform(),
            memory=SQLiteMemoryStore(s.db.url), safety=SafetyClassifier(),
            settings=s.agent,
        )
        await agent.run(task_id, body.instruction)

    background_tasks.add_task(run_task)
    return TaskResponse(task_id=task_id, status="accepted",
        message=f"Poll GET /v1/tasks/{task_id} for status.")


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> dict:
    steps = await memory.get_task_history(task_id)
    if not steps:
        raise HTTPException(404, detail=f"Task {task_id!r} not found")
    return {
        "task_id":     task_id,
        "total_steps": len(steps),
        "steps": [
            {"index": s.step_index,
             "action": s.action_taken.type if s.action_taken else None,
             "success": s.action_success, "error": s.error}
            for s in steps
        ],
    }
