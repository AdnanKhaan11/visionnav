"""
Prompt templates for LLM-based annotation.

Keeping templates in one file means:
  - Easy to improve without touching business logic
  - Version-controlled independently
  - A/B testable (swap templates, compare annotation quality)

Template design principles:
  1. Be specific about format — LLM must return structured output
  2. Reference concrete screen elements — prevents hallucination
  3. Ask for reasoning BEFORE conclusion — chain-of-thought
  4. Separate concerns — one template per annotation task
"""

from __future__ import annotations

REASONING_SYSTEM_PROMPT = """
You are an expert GUI automation annotator for VisionNav, an AI agent that controls computer interfaces.

Your job is to write high-quality reasoning annotations for agent actions.
These annotations become training data that teaches future AI models how to think before acting.

CRITICAL RULES:
1. Only reference UI elements that appear in the OCR text provided
2. Never invent elements that are not in the OCR data
3. Write from the agent's perspective ("I can see...", "I need to...")
4. Be specific about coordinates and elements
5. Explain WHY this action, not just WHAT it is
6. Keep reasoning between 2-5 sentences

OUTPUT FORMAT (JSON only, no markdown, no explanation):
{
  "reasoning": "string — the chain-of-thought reasoning",
  "intent": "string — one sentence: what goal does this action serve?",
  "verified": true/false — are all elements mentioned actually in the OCR text?
}
""".strip()


REASONING_USER_TEMPLATE = """
TASK: {task}

CURRENT SCREEN (OCR text detected):
{ocr_text}

ACTION TAKEN:
  Type: {action_type}
  {coordinates_line}
  {text_line}
  Description: {description}

STEP CONTEXT:
  Step number: {step_index}
  Total steps seen so far: {total_steps}
  Previous action: {previous_action}

Write the reasoning annotation for this action.
""".strip()


DIFFICULTY_SYSTEM_PROMPT = """
You are an expert at rating the difficulty of GUI automation tasks.

Rate difficulty on a scale of 1-5:
  1 = Trivial: single obvious action, no decision needed (click visible button)
  2 = Easy: clear sequence, no ambiguity (type text in focused field)
  3 = Medium: requires reading screen state, mild decision (find correct tab)
  4 = Hard: complex navigation, multiple conditions (find email by date filter)
  5 = Expert: multi-step reasoning, error recovery, rare scenarios

OUTPUT FORMAT (JSON only):
{
  "difficulty": integer 1-5,
  "reason": "one sentence explaining the rating"
}
""".strip()


DIFFICULTY_USER_TEMPLATE = """
TASK: {task}
TOTAL STEPS IN TRAJECTORY: {total_steps}
ACTION TYPE: {action_type}
SCREEN COMPLEXITY: {n_ocr_regions} text regions detected
STEP INDEX: {step_index} of {total_steps}
""".strip()


def build_reasoning_prompt(
    task: str,
    ocr_text: str,
    action_type: str,
    step_index: int,
    total_steps: int,
    description: str = "",
    coordinates: list | None = None,
    text: str | None = None,
    previous_action: str = "none",
) -> tuple[str, str]:
    """
    Build system + user prompts for reasoning annotation.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    coords_line = (
        f"Coordinates: [{coordinates[0]:.3f}, {coordinates[1]:.3f}]"
        if coordinates
        else "Coordinates: N/A"
    )
    text_line = f"Text: {text!r}" if text else "Text: N/A"
    ocr_summary = ocr_text[:800] if ocr_text else "No OCR text available"

    user = REASONING_USER_TEMPLATE.format(
        task=task,
        ocr_text=ocr_summary,
        action_type=action_type,
        coordinates_line=coords_line,
        text_line=text_line,
        description=description or "no description",
        step_index=step_index,
        total_steps=total_steps,
        previous_action=previous_action,
    )
    return REASONING_SYSTEM_PROMPT, user


def build_difficulty_prompt(
    task: str,
    total_steps: int,
    action_type: str,
    step_index: int,
    n_ocr_regions: int,
) -> tuple[str, str]:
    """
    Build system + user prompts for difficulty scoring.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    user = DIFFICULTY_USER_TEMPLATE.format(
        task=task,
        total_steps=total_steps,
        action_type=action_type,
        step_index=step_index,
        n_ocr_regions=n_ocr_regions,
    )
    return DIFFICULTY_SYSTEM_PROMPT, user
