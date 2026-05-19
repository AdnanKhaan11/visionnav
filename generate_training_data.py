"""
Generate real training data by recording actual GUI actions.
Records screenshot + action → saves as training sample.
"""

import sys
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "src")

from visionnav.platforms.desktop import DesktopPlatform
from visionnav.perception.ocr import OCREngine
from visionnav.utils.image import save_screenshot
from data_pipeline.formatters import format_sample

# Output directories
SCREENSHOT_DIR = Path("data/training_screenshots")
OUTPUT_FILE = Path("data/instruction_tuning/real_train.jsonl")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


async def record_action(
    platform: DesktopPlatform,
    ocr: OCREngine,
    task: str,
    action_type: str,
    action_value: str | list,
    reasoning: str,
    step_index: int,
) -> dict:
    """Capture screenshot then record one action as training sample."""

    # Capture screenshot BEFORE action
    arr, meta = await platform.capture()

    # Save screenshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_name = f"step_{step_index:03d}_{timestamp}.png"
    img_path = SCREENSHOT_DIR / img_name
    save_screenshot(arr, img_path)

    # Read screen text
    regions = ocr.run(arr)
    ocr_text = " | ".join([r.text for r in regions[:10]])

    print(f"  Step {step_index}: {action_type} — {ocr_text[:50]}...")

    # Format as training sample
    coords = action_value if action_type == "click" else None
    text = action_value if action_type == "type" else None
    key = action_value if action_type == "key" else None

    sample = format_sample(
        image_path=str(img_path),
        task=task,
        action_type=action_type,
        coordinates=coords,
        text=text,
        key=key,
        reasoning=reasoning,
    )

    return sample


async def record_open_notepad(platform, ocr):
    """Record the complete 'Open Notepad' task as training data."""
    print("\nRecording: Open Notepad task")
    import time

    samples = []

    # Step 1 - Press Win+R
    s = await record_action(
        platform,
        ocr,
        task="Open Notepad",
        action_type="key",
        action_value="win+r",
        reasoning=(
            "I need to open Notepad. "
            "The fastest way is Win+R to open Run dialog, "
            "then type notepad."
        ),
        step_index=1,
    )
    samples.append(s)
    await platform.execute_key("win+r")
    time.sleep(1)

    # Step 2 - Type notepad
    s = await record_action(
        platform,
        ocr,
        task="Open Notepad",
        action_type="type",
        action_value="notepad",
        reasoning=(
            "Run dialog is now open. "
            "I will type 'notepad' to open Notepad application."
        ),
        step_index=2,
    )
    samples.append(s)
    await platform.execute_type("notepad")
    time.sleep(0.5)

    # Step 3 - Press Enter
    s = await record_action(
        platform,
        ocr,
        task="Open Notepad",
        action_type="key",
        action_value="enter",
        reasoning=(
            "I have typed notepad in the Run dialog. "
            "Now I press Enter to execute and open Notepad."
        ),
        step_index=3,
    )
    samples.append(s)
    await platform.execute_key("enter")
    time.sleep(2)

    return samples


async def main():
    platform = DesktopPlatform()
    ocr = OCREngine()

    all_samples = []

    print("Starting in 3 seconds...")
    import time

    time.sleep(3)

    # Record tasks
    samples = await record_open_notepad(platform, ocr)
    all_samples.extend(samples)

    # Save all samples
    with open(OUTPUT_FILE, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved {len(all_samples)} real training samples")
    print(f"Screenshots: {SCREENSHOT_DIR}")
    print(f"Training data: {OUTPUT_FILE}")

    # Show summary
    print("\nSample summary:")
    for i, s in enumerate(all_samples):
        assistant = s["conversations"][2]["value"]
        action_line = [l for l in assistant.split("\n") if "type" in l]
        print(f"  Sample {i+1}: {action_line[0] if action_line else 'unknown'}")


asyncio.run(main())
