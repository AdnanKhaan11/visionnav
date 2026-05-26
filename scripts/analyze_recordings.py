"""
Batch quality analysis of all recordings in a directory.

Loads every JSONL recording, runs quality analysis on each,
prints individual reports, and shows a summary at the end.

Usage:
    python scripts/analyze_recordings.py data/recordings/
    python scripts/analyze_recordings.py data/recordings/ --min-score 0.7
    python scripts/analyze_recordings.py data/recordings/ --show-errors-only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "src")

from visionnav.actions.recorder import ActionRecorder
from visionnav.actions.trajectory_analyzer import TrajectoryAnalyzer, QualityReport


def analyze_directory(
    recordings_dir: Path,
    min_score: float,
    show_errors_only: bool,
) -> None:
    """
    Find and analyze all JSONL files in the given directory.

    For each file:
      1. Load steps using ActionRecorder.load() (streaming — no RAM issues)
      2. Run TrajectoryAnalyzer.analyze()
      3. Print individual report (unless filtered by --show-errors-only)

    Then print a batch summary.

    Args:
        recordings_dir:   directory containing *.jsonl files
        min_score:        minimum quality score to approve for training
        show_errors_only: if True, only print reports with errors
    """
    # ── Find all JSONL files ───────────────────────────────────────────────
    jsonl_files = sorted(recordings_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"\nNo JSONL files found in: {recordings_dir}")
        print("Run the agent with ActionRecorder to create recordings first.")
        return

    # ── Set up tools ──────────────────────────────────────────────────────
    recorder = ActionRecorder(
        task="",  # task per file, not per analyzer
        output_dir=recordings_dir,
    )
    analyzer = TrajectoryAnalyzer(quality_threshold=min_score)

    # ── Track statistics across all files ─────────────────────────────────
    all_reports: list[QualityReport] = []
    issue_counts: dict[str, int] = defaultdict(int)

    print(f"\nAnalyzing {len(jsonl_files)} recording(s) in {recordings_dir}")
    print("═" * 52)

    # ── Process each file ─────────────────────────────────────────────────
    for jsonl_path in jsonl_files:

        # Load steps — generator, reads line by line
        steps = list(recorder.load(jsonl_path))

        if not steps:
            print(f"\n[SKIP] {jsonl_path.name} — empty file")
            continue

        # Extract session info from first step
        session_id = steps[0].session_id or jsonl_path.stem
        task = steps[0].task or "unknown task"

        # Run quality analysis
        report = analyzer.analyze(steps, session_id=session_id, task=task)
        all_reports.append(report)

        # Count all issue types for summary
        for issue in report.issues:
            issue_counts[issue.issue_type] += 1

        # Print this report unless filtered
        has_errors = len(report.errors) > 0
        should_print = (not show_errors_only) or has_errors

        if should_print:
            print()
            print(analyzer.format_report(report))

    # ── Print batch summary ────────────────────────────────────────────────
    if all_reports:
        _print_summary(all_reports, issue_counts, min_score)


def _print_summary(
    reports: list[QualityReport],
    issue_counts: dict[str, int],
    min_score: float,
) -> None:
    """
    Print aggregate statistics across all analyzed recordings.

    Args:
        reports:      all QualityReport objects produced
        issue_counts: total count of each issue type across all files
        min_score:    the threshold used (for display)
    """
    SEP = "═" * 52

    total = len(reports)
    approved = sum(1 for r in reports if r.use_for_training)
    rejected = total - approved
    avg_score = sum(r.quality_score for r in reports) / total if total > 0 else 0.0

    print()
    print(SEP)
    print("Batch Analysis Summary")
    print(SEP)
    print(f"Recordings analyzed:   {total}")
    print(f"Approved for training: {approved} ({approved/total*100:.0f}%)")
    print(f"Flagged for review:    {rejected} ({rejected/total*100:.0f}%)")
    print(f"Average quality score: {avg_score:.2f}")
    print(f"Approval threshold:    {min_score:.2f}")
    print()

    # ── Issue breakdown ───────────────────────────────────────────────────
    if issue_counts:
        print("Issues breakdown:")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  {issue_type:<30} {count}")
    else:
        print("No issues detected across all recordings.")

    print()

    # ── Score distribution ────────────────────────────────────────────────
    if reports:
        perfect = sum(1 for r in reports if r.quality_score == 1.0)
        good = sum(1 for r in reports if 0.8 <= r.quality_score < 1.0)
        marginal = sum(1 for r in reports if 0.6 <= r.quality_score < 0.8)
        poor = sum(1 for r in reports if r.quality_score < 0.6)

        print("Score distribution:")
        print(f"  Perfect  (1.0):       {perfect}")
        print(f"  Good     (0.8–1.0):   {good}")
        print(f"  Marginal (0.6–0.8):   {marginal}")
        print(f"  Poor     (< 0.6):     {poor}")

    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze recording quality for training data selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_recordings.py data/recordings/
  python scripts/analyze_recordings.py data/recordings/ --min-score 0.7
  python scripts/analyze_recordings.py data/recordings/ --show-errors-only
        """,
    )

    parser.add_argument(
        "recordings_dir",
        type=Path,
        help="Directory containing *.jsonl recording files",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        metavar="SCORE",
        help="Minimum quality score to approve for training (default: 0.6)",
    )
    parser.add_argument(
        "--show-errors-only",
        action="store_true",
        help="Only print reports for recordings with errors",
    )

    args = parser.parse_args()

    # Validate the directory exists
    if not args.recordings_dir.exists():
        print(f"Error: directory not found: {args.recordings_dir}")
        sys.exit(1)

    if not args.recordings_dir.is_dir():
        print(f"Error: not a directory: {args.recordings_dir}")
        sys.exit(1)

    analyze_directory(
        recordings_dir=args.recordings_dir,
        min_score=args.min_score,
        show_errors_only=args.show_errors_only,
    )


if __name__ == "__main__":
    main()
