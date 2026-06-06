"""
scripts/run_pipeline.py

Run the VisionNav Dataset Factory pipeline.

SECURITY: API keys are loaded exclusively from .env file.
          Never pass keys as command-line arguments.
          Never hardcode keys in this file.

SETUP:
    1. Copy .env.example to .env
    2. Fill in real API keys in .env
    3. Run: python scripts/run_pipeline.py

OPTIONS:
    --version      Dataset version string (default: 1.1.0)
    --recordings   Recordings directory (default: data/recordings)
    --output       Training output directory (default: data/training)
    --pipeline-dir Pipeline state directory (default: data/pipeline)
    --reset        Clear dedup index and registry before running
                   (use when reprocessing existing recordings)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Load .env file before any other imports
# This must happen before reading os.environ
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed.")
    print("         Run: pip install python-dotenv --break-system-packages")
    print("         API keys will not be loaded from .env\n")

# IMPROVISED CODE: Add both project root and src/ to path.
# data_pipeline is at project root, visionnav package is inside src/.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VisionNav Dataset Factory Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        default="",
        help="Dataset version (leave empty to auto-generate from date)",
    )
    parser.add_argument(
        "--recordings",
        default="data/recordings",
        help="Directory containing JSONL recording files",
    )
    parser.add_argument(
        "--output",
        default="data/training",
        help="Directory for exported training JSONL",
    )
    parser.add_argument(
        "--pipeline-dir",
        default="data/pipeline",
        help="Directory for registry, dedup index, cache",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear dedup index and registry before running",
    )
    args = parser.parse_args()

    pipeline_dir = Path(args.pipeline_dir)

    # Optional reset — clears dedup so existing recordings can be reprocessed
    if args.reset:
        for fname in (".dedup_index.json", "registry.db", "lineage.jsonl"):
            fpath = pipeline_dir / fname
            if fpath.exists():
                fpath.unlink()
                print(f"[reset] Deleted {fpath}")

    # Load keys from environment — only place they should come from
    groq_key = os.environ.get("GROQ_API_KEY", "")
    google_key = os.environ.get("GEMINI_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    # Report key availability without exposing values
    print("\nProvider availability:")
    print(
        f"  Groq:       {'✓ key loaded' if groq_key       else '✗ GROQ_API_KEY not set'}"
    )
    print(
        f"  Google:     {'✓ key loaded' if google_key     else '✗ GEMINI_API_KEY not set'}"
    )
    print(
        f"  OpenRouter: {'✓ key loaded' if openrouter_key else '✗ OPENROUTER_API_KEY not set'}"
    )

    if not any([groq_key, google_key, openrouter_key]):
        print(
            "\nWARNING: No API keys found. Pipeline will run with description fallback.\n"
            "         Quality scores will be 0.60-0.68 instead of 0.85-0.94.\n"
            "         Add keys to .env to enable real annotation.\n"
        )

    from data_pipeline.pipeline import DatasetPipeline, PipelineConfig

    pipeline = DatasetPipeline(
        PipelineConfig(
            pipeline_dir=pipeline_dir,
            groq_api_key=groq_key,
            google_api_key=google_key,
            openrouter_api_key=openrouter_key,
        )
    )

    # IMPROVISED CODE: Auto-generate version from date if not specified.
    # This prevents overwriting previous exports.
    # Format: YYYY.MMDD.HHMM  e.g. 2026.0603.1321
    import datetime

    version = args.version
    if not version:
        now = datetime.datetime.now()
        version = (
            f"{now.year}.{now.month:02d}{now.day:02d}.{now.hour:02d}{now.minute:02d}"
        )

    result = pipeline.run(
        recordings_dir=Path(args.recordings),
        output_dir=Path(args.output),
        dataset_version=version,
    )
    print(f"\nDataset version: {version}")

    print(pipeline.reporter.format(result.metrics))


if __name__ == "__main__":
    main()
