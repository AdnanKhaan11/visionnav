"""Unit tests — prompt builder."""
from visionnav.models.prompt import build_prompt


def test_has_system(test_observation):
    msgs = build_prompt(test_observation, "task", [], [])
    assert msgs[0]["role"] == "system"
    assert "<think>" in msgs[0]["content"]

def test_has_image(test_observation):
    msgs  = build_prompt(test_observation, "task", [], [])
    parts = msgs[1]["content"]
    imgs  = [p for p in parts if p.get("type") == "image_url"]
    assert len(imgs) == 1
    assert imgs[0]["image_url"]["url"].startswith("data:image/png;base64,")

def test_task_in_text(test_observation):
    msgs  = build_prompt(test_observation, "Open Chrome", [], [])
    texts = [p["text"] for p in msgs[1]["content"] if p.get("type") == "text"]
    assert any("Open Chrome" in t for t in texts)
