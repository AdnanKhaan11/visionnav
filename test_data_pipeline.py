import sys, json

sys.path.insert(0, "src")

from data_pipeline.formatters import format_sample
from data_pipeline.validators import _validate

print("Testing data pipeline with mock data...")

# Create 5 mock training samples
mock_samples = [
    {
        "image": "screenshots/screen_001.png",
        "task": "Open Notepad",
        "action_type": "key",
        "key": "win+r",
        "reasoning": "I need to open Notepad. Press Win+R first.",
    },
    {
        "image": "screenshots/screen_002.png",
        "task": "Type Hello World",
        "action_type": "type",
        "text": "Hello World",
        "reasoning": "I see Notepad is open. I will type the text.",
    },
    {
        "image": "screenshots/screen_003.png",
        "task": "Close the window",
        "action_type": "key",
        "reasoning": "Task done. Close with Alt+F4.",
    },
    {
        "image": "screenshots/screen_004.png",
        "task": "Open Chrome",
        "action_type": "key",
        "reasoning": "Need to open Chrome. Use Win+R and type chrome.",
    },
    {
        "image": "screenshots/screen_005.png",
        "task": "Click the search bar",
        "action_type": "click",
        "coordinates": [0.5, 0.08],
        "reasoning": "I can see Chrome is open. Search bar is at top.",
    },
]

# Format each sample
formatted = []
for s in mock_samples:
    sample = format_sample(
        image_path=s["image"],
        task=s["task"],
        action_type=s["action_type"],
        coordinates=s.get("coordinates"),
        text=s.get("text"),
        key=s.get("key"),
        reasoning=s["reasoning"],
    )
    formatted.append(sample)

print(f"Created {len(formatted)} training samples")

# Validate each sample
print("\nValidating samples...")
errors = 0
for i, s in enumerate(formatted):
    try:
        _validate(s)
        print(f"  Sample {i+1}: VALID ✅")
    except AssertionError as e:
        print(f"  Sample {i+1}: INVALID ❌ — {e}")
        errors += 1

# Save to file
import os

os.makedirs("data/instruction_tuning", exist_ok=True)
output_path = "data/instruction_tuning/mock_train.jsonl"

with open(output_path, "w") as f:
    for s in formatted:
        f.write(json.dumps(s) + "\n")

print(f"\nSaved to {output_path}")
print(f"Total: {len(formatted)} samples, {errors} errors")

# Show one sample
print("\nExample sample:")
print(json.dumps(formatted[0], indent=2))
