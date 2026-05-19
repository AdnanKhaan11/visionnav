import sys

sys.path.insert(0, "src")

from data_pipeline.formatters import format_sample
import json

print("Testing data formatter...")

# Create one training sample
sample = format_sample(
    image_path="data/raw/test_screen.png",
    task="Open Notepad",
    action_type="key",
    coordinates=None,
    text=None,
    reasoning="I need to open Notepad. I will press Win+R to open Run dialog.",
)

print("Sample created successfully!")
print()
print("Structure:")
print(json.dumps(sample, indent=2))
