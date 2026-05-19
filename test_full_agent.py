import sys
import asyncio
import pygetwindow as gw
import time

sys.path.insert(0, "src")

from visionnav.platforms.desktop import DesktopPlatform
from visionnav.perception.ocr import OCREngine
from visionnav.actions.verifier import ActionVerifier
from visionnav.actions.schema import Action, ActionType
import time


async def test_task(task_name: str, steps: list):
    print(f"\n{'='*50}")
    print(f"Task: {task_name}")
    print(f"{'='*50}")

    p = DesktopPlatform()
    ocr = OCREngine()
    verifier = ActionVerifier()

    print("Starting in 3 seconds...")
    time.sleep(3)

    for i, step in enumerate(steps):
        print(f"\nStep {i+1}: {step['description']}")

        # Capture before
        before, _ = await p.capture()

        if step["action"] == "focus":
            # import pygetwindow as gw

            windows = gw.getWindowsWithTitle(step["value"])
            if windows:
                windows[0].activate()
                time.sleep(0.5)
                print(f"  Focused: {windows[0].title}")
            else:
                print(f"  Window not found: {step['value']}")

        # Execute action
        if step["action"] == "key":
            await p.execute_key(step["value"])
        elif step["action"] == "type":
            await p.execute_type(step["value"])
        elif step["action"] == "click":
            w, h = p.get_screen_size()
            x = int(step["value"][0] * w)
            y = int(step["value"][1] * h)
            await p.execute_click(x, y)

        time.sleep(step.get("wait", 1))

        # Capture after
        after, meta = await p.capture()

        # Verify
        action = Action(type=ActionType.KEY)
        success, ratio = verifier.verify(before, after, action)
        print(f"  Screen changed: {success} (ratio: {ratio:.3f})")

        # Read screen
        regions = ocr.run(after)
        texts = [r.text for r in regions]
        print(f"  Screen shows: {texts[:5]}")

    print(f"\nTask '{task_name}' complete!")


async def main():
    # Test 1 - Open Notepad
    await test_task(
        "Open Notepad",
        [
            {
                "action": "key",
                "value": "win+r",
                "description": "Open Run dialog",
                "wait": 1,
            },
            {
                "action": "type",
                "value": "notepad",
                "description": "Type notepad",
                "wait": 0.5,
            },
            {
                "action": "key",
                "value": "enter",
                "description": "Press Enter",
                "wait": 2,
            },
        ],
    )

    time.sleep(2)

    # Test 2 - Type in Notepad
    await test_task(
        "Type in Notepad",
        [
            {
                "action": "focus",
                "value": "Notepad",
                "description": "Focus Notepad window",
                "wait": 1,
            },
            {
                "action": "type",
                "value": "Hello from VisionNav!",
                "description": "Type message",
                "wait": 1,
            },
        ],
    )

    # Test 3 - Close Notepad
    await test_task(
        "Close Notepad",
        [
            {
                "action": "focus",
                "value": "Notepad",
                "description": "Focus Notepad",
                "wait": 1,
            },
            {
                "action": "key",
                "value": "alt+f4",
                "description": "Close Notepad",
                "wait": 1,
            },
            {
                "action": "key",
                "value": "enter",
                "description": "Confirm close",
                "wait": 1,
            },
        ],
    )


asyncio.run(main())
