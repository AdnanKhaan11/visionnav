"""
Data Formatter — converts raw GUI annotations to LLaMA-Factory sharegpt format.

This is the format Qwen2.5-VL expects during fine-tuning.
Every training sample has 3 parts:
  1. System prompt  — tells model who it is
  2. User message   — screenshot + task instruction
  3. Assistant reply — reasoning + action JSON

Reference: TongUI approach (_with_thoughts chain-of-thought augmentation)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

OUTPUT_DIR = Path("data/instruction_tuning")

# ─── System Prompt ────────────────────────────────────────────────────────────
# This is what the model sees at the start of every conversation.
# It tells the model exactly how to behave and what format to output.
SYSTEM_PROMPT = """You are VisionNav, an AI agent that controls computer interfaces.

You receive a screenshot of the current screen and a task to complete.

You MUST respond with exactly two blocks:

<think>
Analyse the screen carefully.
What do you see? What UI elements are visible?
What is the best next single action to make progress on the task?
</think>

<action>
{"type": "ACTION_TYPE", "coordinates": [x, y], "text": "...", "description": "..."}
</action>

ACTION TYPES:
  click        → {"type":"click","coordinates":[x,y],"description":"..."}
  double_click → {"type":"double_click","coordinates":[x,y],"description":"..."}
  right_click  → {"type":"right_click","coordinates":[x,y],"description":"..."}
  type         → {"type":"type","text":"your text here","description":"..."}
  key          → {"type":"key","key":"win+r","description":"..."}
  scroll       → {"type":"scroll","direction":"down","amount":3,"description":"..."}
  wait         → {"type":"wait","duration_ms":2000,"description":"..."}
  done         → {"type":"done","description":"Task completed because..."}
  fail         → {"type":"fail","description":"Cannot complete because..."}

RULES:
  - coordinates are ALWAYS normalized floats between 0.0 and 1.0
  - [0.0, 0.0] = top-left corner of screen
  - [1.0, 1.0] = bottom-right corner of screen
  - [0.5, 0.5] = center of screen
  - always think before acting
  - use done when task is fully complete
  - use fail only when task is truly impossible
"""


# ─── Core Formatter ───────────────────────────────────────────────────────────


def format_sample(
    image_path: str,
    task: str,
    action_type: str,
    coordinates: Optional[list] = None,
    text: Optional[str] = None,
    key: Optional[str] = None,
    direction: Optional[str] = None,
    amount: Optional[int] = None,
    reasoning: str = "",
    description: str = "",
) -> dict:
    """
    Convert one raw annotation into a LLaMA-Factory sharegpt training sample.

    This is the format Qwen2.5-VL reads during fine-tuning.
    Every sample teaches the model one action on one screen.

    Args:
        image_path   : path to the screenshot
        task         : natural language task instruction
        action_type  : click / type / key / scroll / done / fail
        coordinates  : [x, y] normalized 0-1 (for click actions)
        text         : text to type (for type actions)
        key          : key combo (for key actions e.g. "win+r", "ctrl+c")
        direction    : scroll direction "up" or "down"
        amount       : scroll amount (number of scroll steps)
        reasoning    : chain-of-thought explanation (TongUI _with_thoughts approach)
        description  : human readable description of the action

    Returns:
        dict in LLaMA-Factory sharegpt format ready for training
    """

    # Build the action JSON
    action: dict = {"type": action_type}

    if coordinates is not None:
        # Validate coordinates are in [0, 1] range
        x, y = float(coordinates[0]), float(coordinates[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Coordinates must be in [0,1] range. Got: [{x}, {y}]")
        action["coordinates"] = [round(x, 4), round(y, 4)]

    if text is not None:
        action["text"] = text

    if key is not None:
        action["key"] = key

    if direction is not None:
        action["direction"] = direction
        action["amount"] = amount or 3

    action["description"] = description or f"Performing {action_type}"

    # Build assistant response
    # Format: <think>reasoning</think>\n<action>JSON</action>
    assistant_content = ""

    if reasoning:
        assistant_content += f"<think>\n{reasoning}\n</think>\n"
    else:
        # Default reasoning if none provided
        assistant_content += (
            f"<think>\nI need to {description or action_type} "
            f"to complete the task: {task}\n</think>\n"
        )

    assistant_content += f"<action>\n{json.dumps(action, indent=2)}\n</action>"

    # Build user message
    user_content = (
        f"<image>\n"
        f"Task: {task}\n\n"
        f"What is the next action to complete this task?"
    )

    return {
        "conversations": [
            {"role": "system", "value": SYSTEM_PROMPT},
            {"role": "user", "value": user_content},
            {"role": "assistant", "value": assistant_content},
        ],
        "images": [image_path],
    }


def make_sharegpt_sample(
    image_path: str,
    task: str,
    action_json: str,
    reasoning: str = "",
) -> dict:
    """
    Alternative constructor — takes pre-built action JSON string.
    Use this when you already have a formatted action string.
    """
    content = ""
    if reasoning:
        content += f"<think>\n{reasoning}\n</think>\n"
    content += f"<action>\n{action_json}\n</action>"

    return {
        "conversations": [
            {"role": "system", "value": SYSTEM_PROMPT},
            {"role": "user", "value": f"<image>\nTask: {task}\nNext action?"},
            {"role": "assistant", "value": content},
        ],
        "images": [image_path],
    }


def run_formatting() -> None:
    """
    Master formatting function.
    Reads cleaned annotations → converts → writes JSONL files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("  Formatting samples → LLaMA-Factory sharegpt JSON...")

    # Check if cleaned data exists
    processed = Path("data/processed/annotations")
    if not processed.exists():
        print("  No processed data found. Run --stage clean first.")
        return

    annotations = list(processed.glob("*.json"))
    if not annotations:
        print("  No annotation files found.")
        return

    print(f"  Found {len(annotations)} annotation files")

    stage1_samples = []  # Grounding samples
    stage2_samples = []  # Action prediction samples
    stage3_samples = []  # Multi-step planning samples

    for ann_path in annotations:
        with open(ann_path) as f:
            ann = json.load(f)

        # Determine which stage this sample belongs to
        action_type = ann.get("action", {}).get("type", "click")
        steps = ann.get("total_steps", 1)

        sample = format_sample(
            image_path=ann.get("image_path", ""),
            task=ann.get("task_instruction", ""),
            action_type=action_type,
            coordinates=ann.get("action", {}).get("coordinates"),
            text=ann.get("action", {}).get("text"),
            key=ann.get("action", {}).get("key"),
            reasoning=ann.get("reasoning", ""),
            description=ann.get("action", {}).get("description", ""),
        )

        # Route to correct stage
        if steps == 1:
            stage1_samples.append(sample)
            stage2_samples.append(sample)
        else:
            stage3_samples.append(sample)

    # Write JSONL files
    _write_jsonl(OUTPUT_DIR / "stage1_grounding.jsonl", stage1_samples)
    _write_jsonl(OUTPUT_DIR / "stage2_action.jsonl", stage2_samples)
    _write_jsonl(OUTPUT_DIR / "stage3_planning.jsonl", stage3_samples)

    total = len(stage1_samples) + len(stage3_samples)
    print(f"  Stage 1 (grounding) : {len(stage1_samples)} samples")
    print(f"  Stage 2 (action)    : {len(stage2_samples)} samples")
    print(f"  Stage 3 (planning)  : {len(stage3_samples)} samples")
    print(f"  Total               : {total} samples")
    print("  Formatting complete")


def _write_jsonl(path: Path, samples: list) -> None:
    """Write list of samples to JSONL file."""
    if not samples:
        return
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Written: {path} ({len(samples)} samples)")
