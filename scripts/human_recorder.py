"""
VisionNav Human Demonstration Recorder — Observation-Action Format.

CRITICAL DESIGN: This recorder captures the Observation → Action format
required by modern GUI agent training.

HOW EACH STEP WORKS:
  1. You are looking at the screen in the state you want the model to see.
  2. You press Enter in this terminal.
  3. A countdown begins — during this time, move your cursor onto the target
     UI element AND click back on the target application so the screenshot
     captures it (not this terminal).
  4. Screenshot is taken — this is the OBSERVATION (before-action state).
     Mouse coordinates are captured at this exact moment.
  5. AFTER the screenshot is taken, perform your action.
  6. Return to this terminal and describe what you just did.

WHY THIS ORDER MATTERS:
  The model is trained on (screenshot, task) → action pairs.
  The screenshot must show what the model would see BEFORE deciding to act.
  If the screenshot shows the result of the action, the model learns
  backwards: "when this is already done, do this" — which causes loops.

  Mouse coordinates are captured at screenshot time so they are perfectly
  synchronised with the observation frame. Capturing them after the action
  risks drift as the cursor moves away from the intended target.

OUTPUT FORMAT:
  data/recordings/session_XXXXXXXX.jsonl
  data/recordings/screenshots/XXXXXXXX/step_000.png  (BEFORE action state)
                                        step_001.png
                                        ...

USAGE:
  python scripts/human_recorder.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

SEP = "═" * 60
SEP2 = "─" * 60

VALID_ACTIONS = {
    "click",
    "double_click",
    "right_click",
    "type",
    "key",
    "scroll",
    "wait",
    "done",
}

# Seconds to wait after Enter before taking screenshot.
# User uses this time to switch focus from terminal to target app
# AND position the cursor on the intended target element.
SCREENSHOT_COUNTDOWN = 7


def capture_screenshot(save_path: Path) -> tuple[np.ndarray, dict, tuple[float, float]]:
    """
    Capture the current screen and save to disk.

    Mouse coordinates are read immediately after the screen grab so that
    the recorded position is synchronised with the observation frame.
    The tiny sequencing gap (microseconds) between the two system calls
    is unavoidable without OS-level hooks and is negligible in practice.

    Returns
    -------
    arr       : np.ndarray  — raw RGB pixel data
    meta      : dict        — {"width": int, "height": int}
    norm_pos  : (float, float) — normalised (x, y) in [0, 1] at capture time
    """
    import mss
    from PIL import Image

    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        arr = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(
            shot.height, shot.width, 3
        )
        screen_w, screen_h = shot.width, shot.height

    # Read cursor position as close to the grab as possible.
    try:
        import pyautogui

        cx, cy = pyautogui.position()
        norm_pos = (
            round(max(0.0, min(1.0, cx / screen_w)), 4),
            round(max(0.0, min(1.0, cy / screen_h)), 4),
        )
    except Exception:
        # pyautogui unavailable — fall back to screen centre so the
        # rest of the pipeline never receives a None coordinate.
        norm_pos = (0.5, 0.5)

    img = Image.fromarray(arr)
    img.save(save_path, format="PNG", compress_level=3)
    return arr, {"width": screen_w, "height": screen_h}, norm_pos


def try_focus_last_app() -> None:
    """Best-effort: activate the most recently used non-terminal window."""
    try:
        import pygetwindow as gw

        candidates = [
            w
            for w in gw.getAllWindows()
            if w.title
            and w.visible
            and w.width > 100
            and not any(
                k in w.title.lower()
                for k in (
                    "powershell",
                    "cmd",
                    "terminal",
                    "python",
                    "windows powershell",
                    "command prompt",
                )
            )
        ]
        if candidates:
            candidates[0].activate()
            time.sleep(0.3)
    except Exception:
        pass  # pygetwindow optional — countdown still helps


def countdown_and_capture(
    screenshot_path: Path,
) -> tuple[np.ndarray, dict, tuple[float, float]]:
    """
    Count down to give the user time to:
      • switch focus to the target application, AND
      • position the cursor on the intended UI element.

    The BEFORE-ACTION screenshot and mouse coordinates are captured together
    at the end of the countdown.

    Returns the same three values as capture_screenshot().
    """
    print(f"\n  ┌─ SCREENSHOT IN:", end="", flush=True)
    for i in range(SCREENSHOT_COUNTDOWN, 0, -1):
        print(f" {i}", end="", flush=True)
        time.sleep(1)

    try_focus_last_app()
    print(" — CAPTURING NOW!")
    return capture_screenshot(screenshot_path)


def prompt(label: str, default: str = "") -> str:
    if default:
        val = input(f"  {label} [{default}]: ").strip()
        return val if val else default
    return input(f"  {label}: ").strip()


def record_action_details(step_index: int, captured_pos: tuple[float, float]) -> dict:
    """
    Collect action metadata AFTER the action has been performed.

    Parameters
    ----------
    step_index   : int                  — current step number (unused here,
                                          kept for future logging hooks)
    captured_pos : (float, float)       — normalised (x, y) mouse position
                                          recorded at screenshot-capture time.
                                          Used as the default click coordinate
                                          for pointer actions.
    """
    action_type = prompt(
        "  Action type (click/type/key/scroll/double_click/right_click/done)"
    ).lower()
    while action_type not in VALID_ACTIONS:
        print(f"    Unknown. Choose from: {', '.join(sorted(VALID_ACTIONS))}")
        action_type = prompt("  Action type").lower()

    action: dict = {"type": action_type}

    if action_type in {"click", "double_click", "right_click"}:
        mx, my = captured_pos
        print(f"  └─ Cursor at screenshot time: [{mx}, {my}]")
        use = prompt("  Use this position as click coords?", "y")
        if use.lower() != "n":
            action["coordinates"] = [mx, my]
        else:
            raw = prompt("  Enter coords as 'x y' (e.g. 0.5 0.22)")
            try:
                cx, cy = map(float, raw.split())
                action["coordinates"] = [
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                ]
            except ValueError:
                # Malformed input — fall back to the captured position
                # rather than silently using an arbitrary default.
                print("  ! Invalid input. Falling back to captured position.")
                action["coordinates"] = [mx, my]

    elif action_type == "type":
        action["text"] = prompt("  Text you typed")

    elif action_type == "key":
        action["key"] = prompt("  Key pressed (e.g. win+e, ctrl+s, enter, f2)")

    elif action_type == "scroll":
        action["direction"] = prompt("  Direction (up/down)", "down")
        action["amount"] = 3

    action["description"] = prompt(
        "  Action description (e.g. 'Click the Save button')"
    )
    return action


def main() -> None:
    print(f"\n{SEP}")
    print("VisionNav  —  Human Demonstration Recorder")
    print("OBSERVATION → ACTION FORMAT")
    print(SEP)
    print()
    print("WORKFLOW FOR EACH STEP:")
    print("  1. Navigate to the state you want the model to learn from")
    print("  2. Press Enter here to start the countdown")
    print("  3. During the countdown:")
    print("       a. Click back on your target application")
    print("       b. Move the cursor onto the UI element you intend to act on")
    print("  4. After 'CAPTURING NOW!' appears: perform your action")
    print("  5. Return here and describe what you did")
    print()
    print("  The screenshot AND cursor position are captured together")
    print("  BEFORE the action, ensuring they are perfectly synchronised.")
    print(SEP2)

    task = input("\nTask instruction: ").strip()
    if not task:
        print("Task cannot be empty.")
        sys.exit(1)

    session_id = str(uuid.uuid4())[:8]
    output_dir = Path("data/recordings")
    screenshot_dir = output_dir / "screenshots" / session_id
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"session_{session_id}.jsonl"

    print(f"\n{SEP2}")
    print(f"Session : {session_id}")
    print(f"Task    : {task}")
    print(f"Output  : {output_path}")
    print(f"{SEP2}\n")

    steps = []
    step_index = 0

    while True:
        print(f"\n{'─'*60}")
        print(f"  STEP {step_index:02d}")
        print(f"{'─'*60}")
        print(f"  Make sure the screen shows the state BEFORE your next action.")
        input(f"  Press Enter to start the countdown...")

        # ── Capture BEFORE-action screenshot + cursor position ─────────────
        screenshot_path = screenshot_dir / f"step_{step_index:03d}.png"
        try:
            arr, meta, captured_pos = countdown_and_capture(screenshot_path)
            size_kb = screenshot_path.stat().st_size / 1024
            print(
                f"  ✓ Screenshot saved: {screenshot_path.name} "
                f"({size_kb:.0f}KB, {meta['width']}×{meta['height']})"
            )
            print(f"  ✓ Cursor position at capture: {list(captured_pos)}")
        except Exception as exc:
            print(f"  ✗ Screenshot failed: {exc}")
            print("    Make sure mss is installed: pip install mss")
            retry = input("  Retry this step? (y/n) [y]: ").strip().lower()
            if retry != "n":
                continue
            sys.exit(1)

        # ── User performs action ───────────────────────────────────────────
        print()
        print(f"  ┌─ Screenshot captured.")
        print(f"  │  NOW perform your action.")
        print(f"  └─ When done, return here and describe it.")
        print()

        # ── Collect action description (cursor coords already known) ───────
        action = record_action_details(step_index, captured_pos)

        # Optional OCR hint
        ocr_hint = prompt(
            "  Key text visible in screenshot (optional, Enter to skip)", ""
        )

        step = {
            "step_index": step_index,
            "task": task,
            "session_id": session_id,
            "action": action,
            "screenshot_path": str(screenshot_path.resolve()),
            "cursor_at_capture": list(captured_pos),  # always stored for reference
            "ocr_text": ocr_hint,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        steps.append(step)
        step_index += 1
        print(f"  ✓ Step {step_index - 1} recorded. (Total: {step_index} steps)")

        if action["type"] == "done":
            break

        if step_index >= 25:
            print(f"\n  Maximum 25 steps reached.")
            break

    # ── Save session ───────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        for step in steps:
            f.write(json.dumps(step, ensure_ascii=False) + "\n")

    print(f"\n{SEP}")
    print(f"Session complete!")
    print(f"Steps recorded: {len(steps)}")
    print(f"Saved to:       {output_path}")
    print(f"\nRun pipeline:")
    print(f"  python scripts/run_pipeline.py")
    print(SEP)


if __name__ == "__main__":
    main()
