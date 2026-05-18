"""Prompt builder for Qwen2.5-VL."""
from __future__ import annotations
from visionnav.perception.fusion import Observation

SYSTEM_PROMPT = """You are VisionNav, an AI agent that controls computer interfaces.

Always respond with TWO blocks:

<think>
Analyse the screen. What is visible? Where is the target element?
What is the best next single action to progress the task?
</think>
<action>
{"type": "ACTION_TYPE", "coordinates": [x, y], "text": "...", "description": "..."}
</action>

ACTION_TYPES:
  click, double_click, right_click, type, key, scroll, wait, done, fail

RULES:
- coordinates are normalised floats in [0.0, 1.0]  (0,0 = top-left, 1,1 = bottom-right)
- always think before acting
- use "done" when the task is fully complete
- use "fail" only when the task is truly impossible
"""


def build_prompt(
    observation: Observation,
    task: str,
    history: list[dict],
    plan: list[str],
) -> list[dict]:
    plan_text    = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan))
    history_text = "\n".join(
        f"  Step {i}: {h.get('content', '')}" for i, h in enumerate(history[-5:])
    ) or "  None yet"

    user_text = (
        f"TASK: {task}\n\n"
        f"PLAN:\n{plan_text}\n\n"
        f"PREVIOUS ACTIONS:\n{history_text}\n\n"
        f"CURRENT SCREEN (OCR):\n{observation.to_text_summary()}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{observation.screenshot_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]
