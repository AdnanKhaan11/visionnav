"""
VisionNav Data Pipeline Master Orchestrator
============================================
Usage:
    python -m data_pipeline.pipeline --stage all
    python -m data_pipeline.pipeline --stage download
    python -m data_pipeline.pipeline --stage clean
    python -m data_pipeline.pipeline --stage ocr
    python -m data_pipeline.pipeline --stage format
    python -m data_pipeline.pipeline --stage validate
"""
from __future__ import annotations
import argparse, sys, time


def run_download() -> None:
    from data_pipeline.downloaders import download_all
    download_all()

def run_clean() -> None:
    from data_pipeline.cleaners import run_cleaning
    run_cleaning()

def run_ocr() -> None:
    from data_pipeline.ocr_enricher import run_ocr_enrichment
    run_ocr_enrichment()

def run_format() -> None:
    from data_pipeline.formatters import run_formatting
    run_formatting()

def run_validate() -> None:
    from data_pipeline.validators import run_validation
    run_validation()


STAGES: dict[str, callable] = {
    "download": run_download,
    "clean":    run_clean,
    "ocr":      run_ocr,
    "format":   run_format,
    "validate": run_validate,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionNav data pipeline")
    parser.add_argument("--stage", choices=[*STAGES.keys(), "all"], default="all")
    args   = parser.parse_args()

    stages = list(STAGES.items()) if args.stage == "all"              else [(args.stage, STAGES[args.stage])]

    print(f"\n{'='*60}")
    print(f"VisionNav Data Pipeline — {len(stages)} stage(s)")
    print(f"{'='*60}\n")

    t0 = time.time()
    for name, fn in stages:
        print(f"[{name.upper()}] Starting...")
        t = time.time()
        try:
            fn()
            print(f"[{name.upper()}] Done in {time.time()-t:.1f}s\n")
        except Exception as exc:
            print(f"[{name.upper()}] FAILED: {exc}", file=sys.stderr)
            sys.exit(1)

    print(f"All stages complete in {time.time()-t0:.1f}s\n")


if __name__ == "__main__":
    main()
