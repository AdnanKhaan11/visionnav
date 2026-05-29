"""
test_pipeline_final.py — complete end-to-end Dataset Factory test.

Creates realistic mock data → runs full pipeline → exports training JSONL.
This is the test that proves the entire Dataset Factory works.

Run: python test_pipeline_final.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "src")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("data/factory_test")
RECORDINGS_DIR = BASE_DIR / "recordings"
OUTPUT_DIR = BASE_DIR / "training_output"
PIPELINE_DIR = BASE_DIR / "pipeline"

for d in [RECORDINGS_DIR, OUTPUT_DIR, PIPELINE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Build realistic mock recordings ───────────────────────────────────────────


def make_screenshot(
    texts: list[tuple[str, tuple[int, int]]],  # (text, (x, y))
    size: tuple[int, int] = (1280, 720),
    bg_color: tuple = (245, 245, 245),
) -> Path:
    """Create a realistic-looking mock screenshot."""
    img = Image.new("RGB", size, color=bg_color)
    draw = ImageDraw.Draw(img)

    # Draw a simple toolbar
    draw.rectangle([0, 0, size[0], 40], fill=(70, 130, 180))
    draw.text((10, 10), "VisionNav Test Browser", fill=(255, 255, 255))

    # Draw content area
    draw.rectangle([0, 40, size[0], size[1]], fill=(255, 255, 255))

    # Draw each text element
    for text, (x, y) in texts:
        draw.text((x, y), text, fill=(30, 30, 30))
        # Draw subtle box around text (simulates UI element)
        tw, th = draw.textbbox((0, 0), text)[2:4]
        draw.rectangle([x - 4, y - 4, x + tw + 4, y + th + 4], outline=(200, 200, 200))

    # Save as PNG with good quality (ensures file > 10KB)
    path = (
        RECORDINGS_DIR
        / f"screenshots"
        / f"mock_{len(list(RECORDINGS_DIR.glob('screenshots/*.png')))}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", compress_level=3)
    return path


def make_session(
    session_id: str,
    task: str,
    steps: list[dict],
) -> Path:
    """Create a JSONL session file with given steps."""
    path = RECORDINGS_DIR / f"session_{session_id}.jsonl"
    with open(path, "w") as f:
        for step in steps:
            f.write(json.dumps(step) + "\n")
    return path


# ── Create Session 1: Email workflow (English) ─────────────────────────────────
ss1 = make_screenshot(
    [
        ("Gmail - Inbox", (50, 80)),
        ("Compose", (50, 150)),
        ("John Smith - Meeting Tomorrow", (50, 200)),
        ("Alice Wang - Project Update", (50, 250)),
        ("Unread (2)", (900, 80)),
        ("Search mail", (300, 55)),
    ],
    size=(1280, 720),
)

ss2 = make_screenshot(
    [
        ("John Smith - Meeting Tomorrow", (50, 80)),
        ("Hi team, our meeting is scheduled for...", (50, 150)),
        ("Reply", (100, 600)),
        ("Forward", (200, 600)),
        ("Archive", (300, 600)),
    ],
    size=(1280, 720),
)

make_session(
    "email_001",
    "Open Gmail and reply to John Smith's email",
    [
        {
            "step_index": 0,
            "task": "Open Gmail and reply to John Smith's email",
            "session_id": "email_001",
            "action": {
                "type": "click",
                "coordinates": [0.04, 0.28],
                "description": "Click John Smith email",
            },
            "screenshot_path": str(ss1),
            "ocr_text": "Gmail Inbox Compose John Smith Meeting Tomorrow Alice Wang Project Update",
            "timestamp": "2026-05-20T10:00:00Z",
        },
        {
            "step_index": 1,
            "task": "Open Gmail and reply to John Smith's email",
            "session_id": "email_001",
            "action": {
                "type": "click",
                "coordinates": [0.08, 0.83],
                "description": "Click Reply button",
            },
            "screenshot_path": str(ss2),
            "ocr_text": "John Smith Meeting Tomorrow Hi team Reply Forward Archive",
            "timestamp": "2026-05-20T10:00:05Z",
        },
        {
            "step_index": 2,
            "task": "Open Gmail and reply to John Smith's email",
            "session_id": "email_001",
            "action": {"type": "done", "description": "Reply window is open"},
            "screenshot_path": str(ss2),
            "ocr_text": "John Smith Meeting Tomorrow Hi team Reply Forward Archive",
            "timestamp": "2026-05-20T10:00:08Z",
        },
    ],
)


# ── Create Session 2: Browser navigation ──────────────────────────────────────
# BEFORE — all browser steps share one screenshot


# AFTER — each step gets its own screenshot
def make_step_screenshot(step_index, session_id, texts, size=(1280, 720)):
    """One unique screenshot per step."""
    img = Image.new("RGB", size, color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], 40], fill=(70, 130, 180))
    draw.text((10, 10), f"Step {step_index} — VisionNav", fill=(255, 255, 255))
    draw.rectangle([0, 40, size[0], size[1]], fill=(255, 255, 255))
    for text, (x, y) in texts:
        draw.text((x, y), text, fill=(30, 30, 30))
    path = RECORDINGS_DIR / "screenshots" / f"{session_id}_step{step_index:03d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", compress_level=3)
    return path


# Then in session creation:
make_session(
    "browser_001",
    "Search for Python tutorials on Google",
    [
        {
            "step_index": 0,
            "screenshot_path": str(
                make_step_screenshot(
                    0,
                    "browser_001",
                    [
                        ("Address bar", (200, 12)),
                        ("https://www.google.com", (300, 12)),
                        ("Search...", (400, 350)),
                        ("Google Search", (450, 420)),
                    ],
                )
            ),
            "task": "Search for Python tutorials on Google",
            "session_id": "browser_001",
            "action": {
                "type": "click",
                "coordinates": [0.21, 0.32],
                "description": "Click search box",
            },
            "ocr_text": "Address bar https google.com Search Google Search Feeling Lucky",
            "timestamp": "2026-05-20T10:01:00Z",
        },
        {
            "step_index": 1,
            "screenshot_path": str(
                make_step_screenshot(
                    1,
                    "browser_001",
                    [
                        ("Address bar", (200, 12)),
                        ("Python tutorials for beginners", (300, 350)),
                        ("Google Search", (450, 420)),
                    ],
                )
            ),
            "task": "Search for Python tutorials on Google",
            "session_id": "browser_001",
            "action": {
                "type": "type",
                "text": "Python tutorials for beginners",
                "description": "Type search query",
            },
            "ocr_text": "Address bar Python tutorials for beginners Google Search",
            "timestamp": "2026-05-20T10:01:03Z",
        },
        {
            "step_index": 2,
            "screenshot_path": str(
                make_step_screenshot(
                    2,
                    "browser_001",
                    [
                        ("Python tutorials for beginners", (200, 60)),
                        ("Google Search", (450, 420)),
                        ("Results loading...", (200, 200)),
                    ],
                )
            ),
            "task": "Search for Python tutorials on Google",
            "session_id": "browser_001",
            "action": {"type": "key", "key": "enter", "description": "Submit search"},
            "ocr_text": "Python tutorials for beginners Google Search Results loading",
            "timestamp": "2026-05-20T10:01:05Z",
        },
        {
            "step_index": 3,
            "screenshot_path": str(
                make_step_screenshot(
                    3,
                    "browser_001",
                    [
                        ("Python Tutorial - W3Schools", (200, 100)),
                        ("Learn Python - Python.org", (200, 180)),
                        ("Python for Beginners - Microsoft", (200, 260)),
                    ],
                )
            ),
            "task": "Search for Python tutorials on Google",
            "session_id": "browser_001",
            "action": {"type": "done", "description": "Search results loaded"},
            "ocr_text": "Python Tutorial W3Schools Learn Python Python.org Beginners Microsoft",
            "timestamp": "2026-05-20T10:01:08Z",
        },
    ],
)

# ── Run the pipeline ───────────────────────────────────────────────────────────
from data_pipeline.pipeline import DatasetPipeline, PipelineConfig

config = PipelineConfig(
    pipeline_dir=PIPELINE_DIR,
    quality_threshold=0.55,  # lower for testing (no real annotation)
    annotate_samples=False,  # no API key in test
)

pipeline = DatasetPipeline(config)

result = pipeline.run(
    recordings_dir=RECORDINGS_DIR,
    output_dir=OUTPUT_DIR,
    dataset_version="1.0.0",
)

# ── Print report ──────────────────────────────────────────────────────────────
print(pipeline.reporter.format(result.metrics))

print(f"\nOutput files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45}  {size_kb:.1f} KB")

print(f"\nRegistry stats:")
stats = pipeline._registry.stats()
for k, v in stats.items():
    if not isinstance(v, dict):
        print(f"  {k:<30} {v}")

print(f"\nDataset versions:")
for v in pipeline._registry.list_versions():
    print(
        f"  v{v['version']}: {v['approved_samples']} approved / {v['total_samples']} total"
    )

# ── Verify output JSONL is valid LLaMA-Factory format ────────────────────────
print(f"\nVerifying export format...")
for jsonl_file in OUTPUT_DIR.glob("*.jsonl"):
    lines = jsonl_file.read_text().strip().split("\n")
    valid = 0
    for line in lines:
        if not line:
            continue
        try:
            obj = json.loads(line)
            assert "conversations" in obj
            assert "images" in obj
            assert len(obj["conversations"]) == 2
            assert obj["conversations"][0]["from"] == "human"
            assert obj["conversations"][1]["from"] == "gpt"
            assert "<think>" in obj["conversations"][1]["value"]
            assert "<action>" in obj["conversations"][1]["value"]
            valid += 1
        except (AssertionError, json.JSONDecodeError) as exc:
            print(f"  ✗ Invalid example in {jsonl_file.name}: {exc}")

    if valid > 0:
        print(f"  ✓ {jsonl_file.name}: {valid} valid training examples")
    else:
        print(
            f"  - {jsonl_file.name}: 0 examples (samples may not meet stage criteria)"
        )
