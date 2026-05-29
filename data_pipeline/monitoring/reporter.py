"""
Pipeline health reporter — formats PipelineMetrics as readable reports.

Produces terminal output that answers at a glance:
  - How many samples processed?
  - Where are we losing samples?
  - What is the quality distribution?
  - Are multilingual samples being collected?
  - Is the export producing enough training data?
"""

from __future__ import annotations

import json
from pathlib import Path

from data_pipeline.monitoring.metrics import PipelineMetrics


class PipelineReporter:
    """
    Formats PipelineMetrics as human-readable text reports.

    Usage:
        reporter = PipelineReporter()
        print(reporter.format(metrics))
        reporter.save(metrics, Path("data/pipeline_reports/run_001.json"))
    """

    SEP = "═" * 60
    SEP2 = "─" * 60

    def format(self, metrics: PipelineMetrics) -> str:
        """
        Format complete pipeline run as a detailed report.
        """
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append(self.SEP)
        lines.append("VisionNav Dataset Factory — Pipeline Report")
        lines.append(f"Run ID:  {metrics.run_id}")
        lines.append(f"Started: {metrics.started_at[:19].replace('T', ' ')}")
        if metrics.finished_at:
            lines.append(f"Ended:   {metrics.finished_at[:19].replace('T', ' ')}")
        lines.append(self.SEP)

        # ── Pipeline funnel ───────────────────────────────────────────────
        lines.append("\nPipeline Funnel (how many samples survived each stage):")
        lines.append(self.SEP2)

        stages = [
            ("Ingestion", metrics.ingestion),
            ("Validation", metrics.validation),
            ("Enrichment", metrics.enrichment),
            ("Annotation", metrics.annotation),
            ("Curation", metrics.curation),
        ]

        max_processed = max(
            (s.processed for _, s in stages if s.processed > 0),
            default=1,
        )

        for stage_name, counter in stages:
            if counter.processed == 0:
                continue
            bar_len = int((counter.passed / max(max_processed, 1)) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            rate = counter.pass_rate * 100
            lines.append(
                f"  {stage_name:<12} {bar}  "
                f"{counter.passed:>5} / {counter.processed:<5}  ({rate:.0f}%)"
            )

        # ── Rejection breakdown ───────────────────────────────────────────
        if metrics.rejection_reasons:
            lines.append(f"\nRejection Reasons:")
            lines.append(self.SEP2)
            sorted_reasons = sorted(
                metrics.rejection_reasons.items(),
                key=lambda x: -x[1],
            )
            for reason, count in sorted_reasons[:10]:
                lines.append(f"  {reason:<35}  {count}")

        # ── Quality distribution ──────────────────────────────────────────
        if metrics.quality_buckets:
            total_scored = sum(metrics.quality_buckets.values())
            lines.append(f"\nQuality Score Distribution ({total_scored} scored):")
            lines.append(self.SEP2)

            bucket_order = [
                "0.90-1.00",
                "0.80-0.90",
                "0.70-0.80",
                "0.60-0.70",
                "0.00-0.60",
            ]
            for bucket in bucket_order:
                count = metrics.quality_buckets.get(bucket, 0)
                if count == 0:
                    continue
                pct = count / max(total_scored, 1) * 100
                bar_len = int(pct / 100 * 25)
                bar = "█" * bar_len
                lines.append(f"  {bucket}  {bar:<25}  {count:>5} ({pct:.0f}%)")

        # ── Language distribution ─────────────────────────────────────────
        if metrics.language_counts:
            total_lang = sum(metrics.language_counts.values())
            lines.append(f"\nLanguage Distribution ({total_lang} detected):")
            lines.append(self.SEP2)

            lang_names = {
                "en": "English",
                "ur": "Urdu   ",
                "ps": "Pashto ",
                "ar": "Arabic ",
                "hi": "Hindi  ",
                "unknown": "Unknown",
            }
            for lang, count in sorted(
                metrics.language_counts.items(), key=lambda x: -x[1]
            ):
                name = lang_names.get(lang, lang.ljust(7))
                pct = count / max(total_lang, 1) * 100
                bar_len = int(pct / 100 * 25)
                bar = "█" * bar_len
                lines.append(f"  {name}  {bar:<25}  {count:>5} ({pct:.0f}%)")

        # ── Training stage distribution ───────────────────────────────────
        if metrics.stage_counts:
            total_staged = sum(metrics.stage_counts.values())
            lines.append(f"\nTraining Stage Assignment ({total_staged} routed):")
            lines.append(self.SEP2)
            for stage, count in sorted(
                metrics.stage_counts.items(), key=lambda x: -x[1]
            ):
                pct = count / max(total_staged, 1) * 100
                lines.append(f"  {stage:<30}  {count:>5} ({pct:.0f}%)")

        # ── Export summary ────────────────────────────────────────────────
        if metrics.exported_per_stage:
            total_exported = sum(metrics.exported_per_stage.values())
            lines.append(f"\nExport Summary ({total_exported} total examples):")
            lines.append(self.SEP2)
            for stage, count in sorted(
                metrics.exported_per_stage.items(), key=lambda x: -x[1]
            ):
                lines.append(f"  {stage:<30}  {count}")

        # ── Final verdict ─────────────────────────────────────────────────
        lines.append(f"\n{self.SEP}")
        total_in = metrics.ingestion.processed
        total_out = (
            sum(metrics.exported_per_stage.values())
            if metrics.exported_per_stage
            else 0
        )
        if total_in > 0:
            overall = total_out / total_in * 100
            lines.append(
                f"Overall yield: {total_out} training examples from "
                f"{total_in} raw steps ({overall:.1f}%)"
            )
        lines.append(self.SEP)

        return "\n".join(lines)

    def save(self, metrics: PipelineMetrics, path: Path) -> None:
        """Save metrics as JSON for storage and future analysis."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
        import structlog

        structlog.get_logger(__name__).info("metrics_saved", path=str(path))
