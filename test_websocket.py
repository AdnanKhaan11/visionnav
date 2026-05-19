"""Test WebSocket streaming."""

import asyncio
import json
import uuid
import httpx
import websockets


async def test_websocket_stream():
    base_url = "http://localhost:8000"
    ws_url = "ws://localhost:8000"

    # Step 1 — Submit a task
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/v1/tasks/",
            json={"instruction": "Open Notepad", "max_steps": 3},
            headers={"authorization": "Bearer dev-key-123"},
        )
        task_id = resp.json()["task_id"]
        print(f"Task submitted: {task_id}")

    # Step 2 — Connect to WebSocket and listen
    ws_endpoint = f"{ws_url}/v1/tasks/{task_id}/stream"
    print(f"Connecting to: {ws_endpoint}")

    async with websockets.connect(ws_endpoint) as ws:
        print("Connected! Waiting for events...\n")
        for _ in range(10):  # Max 10 messages
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                event = json.loads(msg)
                print(f"Event: {event['event']}")
                print(f"  Status : {event.get('status')}")
                print(f"  Step   : {event.get('step')}")
                print(f"  Action : {event.get('action')}")
                print()

                if event["event"] == "task_finished":
                    print("Task finished!")
                    break
            except asyncio.TimeoutError:
                print("Timeout waiting for event")
                break


asyncio.run(test_websocket_stream())
