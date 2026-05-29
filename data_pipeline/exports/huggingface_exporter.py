"""
HuggingFace Dataset exporter.

Converts approved PipelineSamples into a HuggingFace Dataset object
that can be:
  1. Saved locally (Parquet format — efficient, versioned)
  2. Pushed to HuggingFace Hub (private or public)
  3. Loaded by any researcher with one line of code

Why HuggingFace format matters:
  - Industry standard — every serious AI lab uses it
  - Built-in versioning (git-lfs on the Hub)
  - Streaming support for large datasets
  - Automatic dataset cards
  - Integration with Trainer, TRL, LLaMA-Factory

Dataset structure on the Hub:
  visionnav/gui-trajectories-v1/
    README.md               ← dataset card (auto-generated)
    data/
      train-00000-of-00001.parquet
    metadata.json

Loading after push:
  from datasets import load_dataset
  ds = load_dataset("your-org/visionnav-gui-v1", split="train")

Private dataset (for commercial protection):
  ds.push_to_hub("your-org/visionnav-gui-v1", private=True, token="hf_...")
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import structlog

from data_pipeline.core.schemas import PipelineSample, TrainingStage

log = structlog.get_logger(__name__)


@dataclass
class HuggingFaceExportResult:
    """Summary of HuggingFace export."""

    total_samples: int
    exported: int
    skipped: int
    local_path: Path | None = None
    hub_repo_id: str = ""
    hub_url: str = ""


class HuggingFaceExporter:
    """
    Exports PipelineSamples to HuggingFace Datasets format.

    Two output modes:
      1. Local Parquet (always runs — no internet needed)
      2. Hub push (optional — needs HF token)

    Dataset schema per row:
      sample_id:       str   — unique identifier
      session_id:      str   — which recording session
      task:            str   — instruction ("Open Gmail...")
      action_type:     str   — "click", "type", "key", etc.
      coordinates:     str   — "[0.5, 0.22]" or ""
      text_input:      str   — text for TYPE actions
      key_input:       str   — key for KEY actions
      description:     str   — action description
      reasoning:       str   — chain-of-thought (may be empty before annotation)
      intent:          str   — high-level goal of this action
      screenshot:      Image — PIL Image object (actual pixel data)
      ocr_text:        str   — raw OCR string
      detected_lang:   str   — "en", "ur", "ps", etc.
      difficulty:      int   — 1-5
      quality_score:   float — 0.0-1.0
      training_stage:  str   — "stage1_grounding", etc.
      source_type:     str   — "human_demo", "synthetic", etc.
      platform:        str   — "windows", "web", etc.
      tags:            str   — JSON array string
      dataset_version: str   — "1.0.0"
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_local(
        self,
        samples: list[PipelineSample],
        dataset_version: str = "1.0.0",
        split: str = "train",
    ) -> HuggingFaceExportResult:
        """
        Save dataset to local Parquet files.
        Does not require HuggingFace credentials.

        Args:
            samples:         approved PipelineSamples
            dataset_version: semantic version
            split:           "train", "validation", or "test"

        Returns:
            HuggingFaceExportResult with path and stats
        """
        try:
            from datasets import Dataset, Features, Value, Image as HFImage
        except ImportError:
            raise ImportError(
                "HuggingFace datasets library not installed. "
                "Run: pip install datasets"
            )

        rows = []
        skipped = 0

        for sample in samples:
            row = self._sample_to_row(sample)
            if row is None:
                skipped += 1
                continue
            rows.append(row)

        if not rows:
            log.warning("no_rows_to_export", total=len(samples), skipped=skipped)
            return HuggingFaceExportResult(
                total_samples=len(samples),
                exported=0,
                skipped=skipped,
            )

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(rows)

        # Save as Parquet (efficient columnar storage)
        version_clean = dataset_version.replace(".", "_")
        output_path = self._output_dir / f"visionnav_{split}_{version_clean}"

        dataset.save_to_disk(str(output_path))

        # Also save as single Parquet file for easy sharing
        parquet_path = self._output_dir / f"visionnav_{split}_{version_clean}.parquet"
        dataset.to_parquet(str(parquet_path))

        log.info(
            "hf_export_local_complete",
            exported=len(rows),
            skipped=skipped,
            parquet=str(parquet_path),
            arrow=str(output_path),
        )

        # Write dataset card
        self._write_dataset_card(
            output_dir=output_path,
            n_samples=len(rows),
            dataset_version=dataset_version,
            split=split,
            samples=samples,
        )

        return HuggingFaceExportResult(
            total_samples=len(samples),
            exported=len(rows),
            skipped=skipped,
            local_path=output_path,
        )

    def push_to_hub(
        self,
        samples: list[PipelineSample],
        repo_id: str,
        hf_token: str,
        dataset_version: str = "1.0.0",
        private: bool = True,
        split: str = "train",
    ) -> HuggingFaceExportResult:
        """
        Export and push directly to HuggingFace Hub.

        Args:
            samples:         approved PipelineSamples
            repo_id:         "your-org/visionnav-gui-v1"
            hf_token:        HuggingFace token (settings.huggingface_token)
            dataset_version: semantic version
            private:         True = private repo (recommended for production)
            split:           dataset split name

        Returns:
            HuggingFaceExportResult with hub URL
        """
        # First export locally
        result = self.export_local(samples, dataset_version, split)

        if result.exported == 0:
            return result

        try:
            from datasets import load_from_disk
        except ImportError:
            raise ImportError("Run: pip install datasets")

        dataset = load_from_disk(str(result.local_path))

        log.info(
            "pushing_to_hub",
            repo_id=repo_id,
            private=private,
            samples=result.exported,
        )

        dataset.push_to_hub(
            repo_id=repo_id,
            token=hf_token,
            private=private,
            split=split,
            commit_message=f"Dataset Factory v{dataset_version} — {result.exported} samples",
        )

        hub_url = f"https://huggingface.co/datasets/{repo_id}"
        log.info("hub_push_complete", url=hub_url)

        result.hub_repo_id = repo_id
        result.hub_url = hub_url
        return result

    def _sample_to_row(self, sample: PipelineSample) -> dict | None:
        """
        Convert one PipelineSample to a flat dict row.
        Returns None if sample cannot be converted (screenshot missing).
        """
        if sample.raw is None:
            return None

        # Load screenshot as PIL Image
        screenshot_img = self._load_image(sample.raw.screenshot_path)
        if screenshot_img is None:
            return None

        # Coordinates as string (HF datasets doesn't have List[float] type easily)
        coords_str = ""
        if sample.raw.coordinates:
            coords_str = json.dumps(sample.raw.coordinates)

        # Annotation fields
        reasoning = ""
        intent = ""
        difficulty = 0
        if sample.annotation:
            reasoning = sample.annotation.reasoning or ""
            intent = sample.annotation.intent or ""
            difficulty = sample.annotation.difficulty or 0

        # Quality fields
        quality_score = 0.0
        if sample.quality:
            quality_score = sample.quality.quality_score

        # Enrichment fields
        detected_lang = "unknown"
        platform = "unknown"
        if sample.enriched:
            detected_lang = sample.enriched.detected_language
            platform = sample.enriched.detected_platform

        return {
            "sample_id": sample.sample_id,
            "session_id": sample.session_id,
            "step_index": sample.step_index,
            "task": sample.task,
            "task_category": sample.task_category,
            "action_type": sample.raw.action_type,
            "coordinates": coords_str,
            "text_input": sample.raw.text or "",
            "key_input": sample.raw.key or "",
            "description": sample.raw.description or "",
            "reasoning": reasoning,
            "intent": intent,
            "screenshot": screenshot_img,  # PIL Image — HF handles this
            "ocr_text": sample.raw.ocr_text_raw or "",
            "detected_lang": detected_lang,
            "platform": platform,
            "difficulty": difficulty,
            "quality_score": round(quality_score, 4),
            "training_stage": sample.training_stage or "",
            "source_type": sample.source_type,
            "tags": json.dumps(sample.tags),
            "dataset_version": sample.dataset_version or "",
        }

    def _load_image(self, path: str):
        """Load screenshot as PIL Image. Returns None if file missing."""
        try:
            from PIL import Image

            img_path = Path(path)
            if not img_path.exists():
                return None
            return Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("image_load_failed", path=path, error=str(exc))
            return None

    def _write_dataset_card(
        self,
        output_dir: Path,
        n_samples: int,
        dataset_version: str,
        split: str,
        samples: list[PipelineSample],
    ) -> None:
        """
        Write README.md dataset card to the output directory.
        HuggingFace Hub displays this as the dataset documentation.
        """
        # Count languages
        lang_counts: dict[str, int] = {}
        for s in samples:
            if s.enriched:
                lang = s.enriched.detected_language
                lang_counts[lang] = lang_counts.get(lang, 0) + 1

        lang_table = "\n".join(
            f"| {lang} | {count} |"
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1])
        )

        card = f"""---
license: apache-2.0
task_categories:
  - image-to-text
  - visual-question-answering
language:
  - en
  - ur
  - ps
tags:
  - gui-automation
  - agent
  - vlm
  - visionnav
  - computer-use
pretty_name: VisionNav GUI Trajectories v{dataset_version}
size_categories:
  - {"1K<n<10K" if n_samples < 10000 else "10K<n<100K"}
---

# VisionNav GUI Trajectories v{dataset_version}

GUI automation trajectory dataset for training VisionNav agents.

## Dataset Description

VisionNav-GUI contains annotated human demonstrations of desktop and web GUI interactions.
Each sample is one step in a task trajectory: a screenshot + action + reasoning chain.

This dataset powers the VisionNav AI agent — an AI system that controls computer
interfaces through visual understanding and reasoning.

## Dataset Statistics

- **Total samples**: {n_samples}
- **Version**: {dataset_version}
- **Split**: {split}

## Language Distribution

| Language | Samples |
|----------|---------|
{lang_table}

## Schema

| Field | Type | Description |
|-------|------|-------------|
| sample_id | string | Unique sample identifier |
| task | string | Natural language task instruction |
| action_type | string | click / type / key / scroll / done |
| coordinates | string | JSON [x, y] normalized to [0,1] |
| screenshot | image | Screen state before the action |
| reasoning | string | Chain-of-thought before action |
| ocr_text | string | Text detected on screen |
| detected_lang | string | UI language (en/ur/ps/ar) |
| difficulty | int | 1=trivial to 5=expert |
| quality_score | float | 0.0-1.0 pipeline quality score |
| training_stage | string | stage1_grounding / stage2_action / stage3_planning |

## Usage

```python
from datasets import load_dataset

# Load from Hub
ds = load_dataset("your-org/visionnav-gui-v{dataset_version}", split="train")

# Access a sample
sample = ds[0]
print(sample["task"])
sample["screenshot"].show()

# Filter by language
urdu = ds.filter(lambda x: x["detected_lang"] == "ur")

# Filter by training stage
stage2 = ds.filter(lambda x: x["training_stage"] == "stage2_action")
```

## License

Apache 2.0 — see LICENSE file.

## Citation

```bibtex
@dataset{{visionnav2026,
  title={{VisionNav GUI Trajectories}},
  year={{2026}},
  version={{{dataset_version}}},
  url={{https://huggingface.co/datasets/your-org/visionnav-gui-v{dataset_version}}}
}}
```
"""
        readme_path = output_dir / "README.md"
        readme_path.write_text(card, encoding="utf-8")
        log.info("dataset_card_written", path=str(readme_path))
