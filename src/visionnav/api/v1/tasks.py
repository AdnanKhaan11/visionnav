"""Task management endpoints."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel

from visionnav.api.dependencies import get_memory, verify_api_key
from visionnav.settings import Settings

router = APIRouter(prefix="/tasks", tags=["tasks"])


# ─── Request / Response Models ────────────────────────────────────────────────


class TaskRequest(BaseModel):
    instruction: str
    platform: str = "desktop"
    max_steps: int = 50


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _determine_status(steps: list) -> str:
    """
    Determine task status from step history.
    Centralised here so all endpoints use same logic.
    """
    if not steps:
        return "running"

    last = steps[-1]
    last_type = last.action_taken.type if last.action_taken else None

    if last_type == "done":
        return "completed"
    if last_type == "fail":
        return "failed"
    if last.error:
        return "failed"
    return "running"


async def _build_agent():
    """
    Build a fully configured VisionNavAgent.
    Centralised here so submit_task and future endpoints
    always use identical agent configuration.
    Future: inject model backend via settings.
    """
    from visionnav.agent.agent import VisionNavAgent
    from visionnav.memory.sqlite import SQLiteMemoryStore
    from visionnav.models.local import LocalModelBackend
    from visionnav.platforms.desktop import DesktopPlatform
    from visionnav.safety.classifier import SafetyClassifier

    s = Settings()
    return VisionNavAgent(
        model=LocalModelBackend(s.model),
        platform=DesktopPlatform(),
        memory=SQLiteMemoryStore(s.db.url),
        safety=SafetyClassifier(),
        settings=s.agent,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/", response_model=TaskResponse, status_code=202)
async def submit_task(
    body: TaskRequest,
    background_tasks: BackgroundTasks,
    memory=Depends(get_memory),
    _key: str = Depends(verify_api_key),
) -> TaskResponse:
    """
    Submit a task for the agent to execute.

    Returns 202 immediately with task_id.
    Agent runs in background.

    Two ways to track progress:
      1. Poll GET /v1/tasks/{task_id}
      2. Stream WS  /v1/tasks/{task_id}/stream
    """
    task_id = str(uuid.uuid4())

    async def run_task() -> None:
        agent = await _build_agent()
        await agent.run(task_id, body.instruction)

    background_tasks.add_task(run_task)

    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message=f"Task accepted. Stream at WS /v1/tasks/{task_id}/stream",
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
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id!r} not found",
        )

    return {
        "task_id": task_id,
        "status": _determine_status(steps),
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
    Note: if agent is mid-step it finishes that step first.
    """
    steps = await memory.get_task_history(task_id)

    if not steps:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id!r} not found",
        )

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
    One screenshot captured per step.
    """
    steps = await memory.get_task_history(task_id)

    if not steps:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id!r} not found",
        )

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


@router.websocket("/{task_id}/stream")
async def stream_task(
    task_id: str,
    websocket: WebSocket,
    memory=Depends(get_memory),
) -> None:
    """
    WebSocket endpoint — stream live task updates.

    Connect:
        ws://localhost:8000/v1/tasks/{task_id}/stream

    Receives JSON events every second:
        {event: "step_update", status: "running", step: 0, action: "click"}
        {event: "task_finished", status: "completed"}

    Browser usage:
        const ws = new WebSocket('ws://localhost:8000/v1/tasks/abc/stream')
        ws.onmessage = e => console.log(JSON.parse(e.data))
    """
    await websocket.accept()

    try:
        last_step_index = -1

        while True:
            steps = await memory.get_task_history(task_id)

            # Task not started yet — keep waiting
            if not steps:
                await websocket.send_json(
                    {
                        "event": "waiting",
                        "task_id": task_id,
                        "message": "Task not started yet",
                    }
                )
                await asyncio.sleep(1)
                continue

            latest = steps[-1]
            status = _determine_status(steps)
            action_type = latest.action_taken.type if latest.action_taken else None

            # Only send update when something new happened
            if latest.step_index != last_step_index:
                last_step_index = latest.step_index

                await websocket.send_json(
                    {
                        "event": "step_update",
                        "task_id": task_id,
                        "status": status,
                        "step": latest.step_index,
                        "action": action_type,
                        "success": latest.action_success,
                        "error": latest.error,
                        "reasoning": latest.reasoning[:100] if latest.reasoning else "",
                        "screenshot": latest.screenshot_path,
                    }
                )

            # Stop streaming when task is finished
            if status in ("completed", "failed"):
                await websocket.send_json(
                    {
                        "event": "task_finished",
                        "task_id": task_id,
                        "status": status,
                    }
                )
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        # Client disconnected — clean exit
        pass
