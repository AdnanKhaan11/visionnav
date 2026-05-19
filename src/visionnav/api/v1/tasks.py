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
    platform: str = "desktop"
    max_steps: int = 50


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


@router.post("/", response_model=TaskResponse, status_code=202)
async def submit_task(
    body: TaskRequest,
    background_tasks: BackgroundTasks,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> TaskResponse:
    """
    Submit a task for the agent to execute.
    Returns immediately with task_id.
    Poll GET /v1/tasks/{task_id} for status.
    """
    task_id = str(uuid.uuid4())

    async def run_task() -> None:
        from visionnav.agent.agent import VisionNavAgent
        from visionnav.memory.sqlite import SQLiteMemoryStore
        from visionnav.models.local import LocalModelBackend
        from visionnav.platforms.desktop import DesktopPlatform
        from visionnav.safety.classifier import SafetyClassifier

        s = Settings()
        agent = VisionNavAgent(
            model=LocalModelBackend(s.model),
            platform=DesktopPlatform(),
            memory=SQLiteMemoryStore(s.db.url),
            safety=SafetyClassifier(),
            settings=s.agent,
        )
        await agent.run(task_id, body.instruction)

    background_tasks.add_task(run_task)
    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message=f"Poll GET /v1/tasks/{task_id} for status.",
    )


@router.get("/{task_id}")
async def get_task(
    task_id: str,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> dict:
    """
    Get task status and full execution history.

    Status values:
      running   → agent is still working
      completed → agent finished successfully
      failed    → agent could not complete task
    """
    steps = await memory.get_task_history(task_id)

    if not steps:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    # Determine current status from last step
    last = steps[-1]
    last_type = last.action_taken.type if last.action_taken else None

    if last_type == "done":
        status = "completed"
    elif last_type == "fail":
        status = "failed"
    elif last.error:
        status = "failed"
    else:
        status = "running"

    return {
        "task_id": task_id,
        "status": status,
        "total_steps": len(steps),
        "steps": [
            {
                "index": s.step_index,
                "action": s.action_taken.type if s.action_taken else None,
                "success": s.action_success,
                "error": s.error,
            }
            for s in steps
        ],
    }


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> dict:
    """
    Cancel a running task.
    Marks task as failed in database.
    Note: if agent is mid-step it will finish that step first.
    """
    steps = await memory.get_task_history(task_id)

    if not steps:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    await memory.mark_task_complete(
        task_id,
        success=False,
        result="Cancelled by user",
    )

    return {
        "task_id": task_id,
        "status": "cancelled",
        "message": "Task has been cancelled",
    }


@router.get("/{task_id}/screenshots")
async def get_screenshots(
    task_id: str,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> dict:
    """
    Get list of screenshots taken during task execution.
    Each step captures one screenshot.
    """
    steps = await memory.get_task_history(task_id)

    if not steps:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    screenshots = [
        {
            "step": s.step_index,
            "path": s.screenshot_path,
            "action": s.action_taken.type if s.action_taken else None,
        }
        for s in steps
        if s.screenshot_path
    ]

    return {
        "task_id": task_id,
        "total": len(screenshots),
        "screenshots": screenshots,
    }
