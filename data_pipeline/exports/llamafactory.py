"""
LLaMA-Factory exporter — converts approved PipelineSamples to training format.

LLaMA-Factory expects ShareGPT format for VLM fine-tuning:

    {
      "conversations": [
        {"from": "human", "value": "<image>\nTask: ...\nScreen: ...\nWhat next?"},
        {"from": "gpt",   "value": "<think>...</think>\n<action>...</action>"}
      ],
      "images": ["absolute/path/to/screenshot.png"]
    }

One JSON object per line in the output JSONL.
The dataset_info.json tells LLaMA-Factory where the file is.

Three stage formats:
  Stage 1 (grounding):  focus on element location
  Stage 2 (action):     focus on action prediction
  Stage 3 (planning):   focus on multi-step reasoning
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from data_pipeline.core.exceptions import ExportError
from data_pipeline.core.schemas import PipelineSample, TrainingStage

log = structlog.get_logger(__name__)


# ── System prompts per stage ───────────────────────────────────────────────────

_SYSTEM_PROMPTS = {
    TrainingStage.STAGE1_GROUNDING.value: (
        "You are VisionNav, an AI agent that controls computer interfaces. "
        "Given a screenshot and a target element, output the normalized "
        "coordinates [x, y] where x and y are between 0.0 and 1.0. "
        "Think step by step inside <think>...</think> tags, "
        "then output your action inside <action>...</action> as JSON."
    ),
    TrainingStage.STAGE2_ACTION.value: (
        "You are VisionNav, an AI agent that controls computer interfaces. "
        "Given a screenshot, a task, and screen content, decide the single "
        "best next action to take. "
        "Think step by step inside <think>...</think> tags, "
        "then output your action inside <action>...</action> as JSON."
    ),
    TrainingStage.STAGE3_PLANNING.value: (
        "You are VisionNav, an AI agent that controls computer interfaces. "
        "Given a screenshot, a multi-step task, previous actions, and screen "
        "content, reason about the overall plan and decide the best next action. "
        "Think step by step inside <think>...</think> tags, "
        "then output your action inside <action>...</action> as JSON."
    ),
}

_DEFAULT_SYSTEM = _SYSTEM_PROMPTS[TrainingStage.STAGE2_ACTION.value]


# ── Dataclass for one training example ────────────────────────────────────────


@dataclass
class TrainingExample:
    """
    One LLaMA-Factory ShareGPT training example.
    Directly serializable to the expected JSONL format.
    """

    conversations: list[dict]  # [{"from": "human", "value": ...}, {"from": "gpt", ...}]
    images: list[str]  # absolute paths to screenshot files
    system: str = ""  # system prompt (per stage)

    def to_dict(self) -> dict:
        d: dict = {"conversations": self.conversations, "images": self.images}
        if self.system:
            d["system"] = self.system
        return d


@dataclass
class ExportResult:
    """Summary of one export run."""

    output_path: Path
    stage: str
    total_samples: int
    exported: int
    skipped: int
    reasons_skipped: dict[str, int] = field(default_factory=dict)

    @property
    def export_rate(self) -> float:
        return self.exported / max(self.total_samples, 1)


# ── Exporter ──────────────────────────────────────────────────────────────────


class LlamaFactoryExporter:
    """
    Exports approved PipelineSamples to LLaMA-Factory training format.

    Usage:
        exporter = LlamaFactoryExporter(output_dir=Path("data/training"))
        result   = exporter.export(samples, stage=TrainingStage.STAGE2_ACTION)
        print(f"Exported {result.exported} samples to {result.output_path}")

    Output files:
        data/training/
          stage2_action_v1.0.0.jsonl      ← training data
          dataset_info.json               ← LLaMA-Factory config
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        samples: list[PipelineSample],
        stage: str,
        dataset_version: str = "1.0.0",
    ) -> ExportResult:
        """
        Export samples for one training stage.

        Only exports samples where:
          - status == approved
          - training_stage matches the requested stage (or ALL_STAGES)
          - screenshot file exists
          - annotation has reasoning

        Args:
            samples:         list of PipelineSamples to export
            stage:           TrainingStage value to export
            dataset_version: used in output filename

        Returns:
            ExportResult with stats and output path
        """
        filename = f"{stage}_{dataset_version.replace('.', '_')}.jsonl"
        output_path = self._output_dir / filename

        skip_reasons: dict[str, int] = {}
        exported = 0
        skipped = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:

                # ── Eligibility checks ────────────────────────────────────
                skip_reason = self._check_eligibility(sample, stage)
                if skip_reason:
                    skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
                    skipped += 1
                    continue

                # ── Build training example ────────────────────────────────
                try:
                    example = self._build_example(sample, stage)
                    f.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
                    exported += 1
                except ExportError as exc:
                    skip_reasons["build_error"] = skip_reasons.get("build_error", 0) + 1
                    skipped += 1
                    log.warning(
                        "export_sample_failed",
                        sample_id=sample.sample_id,
                        error=str(exc),
                    )

        log.info(
            "export_complete",
            stage=stage,
            exported=exported,
            skipped=skipped,
            output=str(output_path),
        )

        # Update dataset_info.json for LLaMA-Factory
        self._update_dataset_info(output_path, stage, dataset_version, exported)

        return ExportResult(
            output_path=output_path,
            stage=stage,
            total_samples=len(samples),
            exported=exported,
            skipped=skipped,
            reasons_skipped=skip_reasons,
        )

    def export_all_stages(
        self,
        samples: list[PipelineSample],
        dataset_version: str = "1.0.0",
    ) -> dict[str, ExportResult]:
        """
        Export samples for all three training stages in one call.
        Returns dict mapping stage name → ExportResult.
        """
        results: dict[str, ExportResult] = {}

        for stage in [
            TrainingStage.STAGE1_GROUNDING.value,
            TrainingStage.STAGE2_ACTION.value,
            TrainingStage.STAGE3_PLANNING.value,
        ]:
            result = self.export(samples, stage, dataset_version)
            results[stage] = result
            log.info(
                "stage_export_done",
                stage=stage,
                exported=result.exported,
            )

        return results

    # ── Private helpers ────────────────────────────────────────────────────

    def _check_eligibility(
        self,
        sample: PipelineSample,
        stage: str,
    ) -> str | None:
        """
        Check if sample is eligible for export.
        Returns None if eligible, or a reason string if not.
        """
        from data_pipeline.core.schemas import SampleStatus

        # Must be approved
        if sample.status != SampleStatus.APPROVED.value:
            return f"status:{sample.status}"

        # Must match requested stage (or be ALL_STAGES)
        if (
            sample.training_stage != stage
            and sample.training_stage != TrainingStage.ALL_STAGES.value
        ):
            return f"wrong_stage:{sample.training_stage}"

        # Must have raw content
        if sample.raw is None:
            return "missing_raw"

        # Must have annotation with reasoning
        if sample.annotation is None or not sample.annotation.reasoning:
            return "missing_reasoning"

        # Screenshot must exist
        if not Path(sample.raw.screenshot_path).exists():
            return "screenshot_missing"

        return None  # eligible

    def _build_example(
        self,
        sample: PipelineSample,
        stage: str,
    ) -> TrainingExample:
        """
        Convert PipelineSample into a TrainingExample.
        """
        if sample.raw is None or sample.annotation is None:
            raise ExportError("sample.raw or sample.annotation is None")

        # ── Build human turn ──────────────────────────────────────────────
        human_value = self._build_human_turn(sample, stage)

        # ── Build assistant turn ──────────────────────────────────────────
        gpt_value = self._build_assistant_turn(sample)

        system = _SYSTEM_PROMPTS.get(stage, _DEFAULT_SYSTEM)

        return TrainingExample(
            conversations=[
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": gpt_value},
            ],
            images=[str(Path(sample.raw.screenshot_path).resolve())],
            system=system,
        )

    def _build_human_turn(
        self,
        sample: PipelineSample,
        stage: str,
    ) -> str:
        """
        Build the human message — what the model receives as input.

        Includes:
          - Image placeholder (<image>)
          - Task instruction
          - Screen content (OCR text with bounding boxes)
          - Step context
        """
        lines: list[str] = ["<image>"]
        lines.append(f"Task: {sample.task}")
        lines.append(f"Step: {sample.step_index}")

        # OCR screen content
        if sample.enriched and sample.enriched.ocr_regions_detailed:
            lines.append("\nScreen content:")
            for region in sample.enriched.ocr_regions_detailed[:30]:
                bbox = region.get("bbox", [])
                text = region.get("text", "")
                if bbox and len(bbox) == 4:
                    lines.append(
                        f"  - '{text}' at "
                        f"[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]"
                    )
        elif sample.raw.ocr_text_raw:
            lines.append(f"\nScreen text: {sample.raw.ocr_text_raw[:300]}")

        # For planning stage, add previous action context
        if stage == TrainingStage.STAGE3_PLANNING.value:
            lines.append(f"\nPrevious action description: {sample.raw.description}")

        lines.append("\nWhat is the next action?")
        return "\n".join(lines)

    def _build_assistant_turn(self, sample: PipelineSample) -> str:
        """
        Build the assistant response — what the model should output.

        Format:
          <think>
          {reasoning}
          </think>
          <action>{"type": "...", ...}</action>
        """
        if sample.raw is None or sample.annotation is None:
            raise ExportError("Cannot build assistant turn: missing raw or annotation")

        reasoning = sample.annotation.reasoning.strip()
        action = self._build_action_json(sample)

        return f"<think>\n{reasoning}\n</think>\n<action>{action}</action>"

    def _build_action_json(self, sample: PipelineSample) -> str:
        """Serialize action to JSON string."""
        if sample.raw is None:
            raise ExportError("sample.raw is None")

        action_dict: dict = {"type": sample.raw.action_type}

        if sample.raw.coordinates is not None:
            action_dict["coordinates"] = [
                round(sample.raw.coordinates[0], 4),
                round(sample.raw.coordinates[1], 4),
            ]
        if sample.raw.text is not None:
            action_dict["text"] = sample.raw.text
        if sample.raw.key is not None:
            action_dict["key"] = sample.raw.key
        if sample.raw.description:
            action_dict["description"] = sample.raw.description

        return json.dumps(action_dict, ensure_ascii=False)

    def _update_dataset_info(
        self,
        output_path: Path,
        stage: str,
        dataset_version: str,
        n_samples: int,
    ) -> None:
        """
        Update dataset_info.json — LLaMA-Factory needs this to find the data.

        LLaMA-Factory reads this file to know:
          - What datasets exist
          - Which file contains each dataset
          - How many samples
        """
        info_path = self._output_dir / "dataset_info.json"

        # Load existing info or start fresh
        if info_path.exists():
            with open(info_path, "r") as f:
                info = json.load(f)
        else:
            info = {}

        dataset_name = f"visionnav_{stage}_{dataset_version.replace('.', '_')}"
        info[dataset_name] = {
            "file_name": output_path.name,
            "formatting": "sharegpt",
            "columns": {
                "prompt": "conversations",
                "images": "images",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system",
            },
            "num_samples": n_samples,
            "stage": stage,
            "version": dataset_version,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        log.info(
            "dataset_info_updated",
            dataset=dataset_name,
            samples=n_samples,
            info_path=str(info_path),
        )
