"""Convert annotations to LLaMA-Factory sharegpt conversation format."""
from __future__ import annotations
from pathlib import Path

OUTPUT_DIR = Path("data/instruction_tuning")

SYSTEM_PROMPT = (
    "You are VisionNav, an AI GUI agent. "
    "Think inside <think>...</think> then output your action inside <action>...</action>."
)


def run_formatting() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("  Formatting samples → LLaMA-Factory sharegpt JSON...")
    # TODO Phase 1: read processed annotations
    # TODO Phase 1: add _with_thoughts chain-of-thought (TongUI approach via GPT-4o)
    # TODO Phase 1: write stage1 / stage2 / stage3 JSONL files
    print("  (stub) Formatting complete")


def make_sharegpt_sample(
    image_path: str,
    task: str,
    action_json: str,
    reasoning: str = "",
) -> dict:
    """One training sample in LLaMA-Factory sharegpt format."""
    content = ""
    if reasoning:
        content += f"<think>\n{reasoning}\n</think>\n"
    content += f"<action>\n{action_json}\n</action>"
    return {
        "conversations": [
            {"role": "system",    "value": SYSTEM_PROMPT},
            {"role": "user",      "value": f"<image>\nTask: {task}\nNext action?"},
            {"role": "assistant", "value": content},
        ],
        "images": [image_path],
    }
