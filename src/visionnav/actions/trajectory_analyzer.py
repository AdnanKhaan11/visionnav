"""
TrajectoryAnalyzer — quality gate for recorded training data.

Before a recording becomes a training sample it must pass quality checks.
This module is the gatekeeper.

"Your model quality cannot exceed the quality of your dataset."

What we check:
  1. Empty trajectory       → ERROR   (nothing to learn from)
  2. Action loops           → WARNING (agent was stuck)
  3. Redundant actions      → WARNING (accidental duplicates)
  4. Low diversity          → WARNING (model learns to repeat one action)
  5. Impossible coordinates → ERROR   (cannot be used for training)

Quality score formula:
  Start at 1.0
  Each ERROR   subtracts 0.3
  Each WARNING subtracts 0.1
  Clamped to [0.0, 1.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from visionnav.actions.recorder import RecordedStep
from visionnav.actions.schema import ActionType

# ─── Issue Types ──────────────────────────────────────────────────────────────


class IssueSeverity(Enum):
    INFO = "info"  # interesting, not a problem
    WARNING = "warning"  # possible problem, review recommended
    ERROR = "error"  # definitely bad, do not use for training


@dataclass(frozen=True)
class TrajectoryIssue:
    """
    One detected quality problem in a trajectory.

    frozen=True means this object cannot be modified after creation.
    Issues are facts — they should not change.
    """

    severity: IssueSeverity
    issue_type: str  # machine-readable: "loop_detected"
    description: str  # human-readable explanation
    step_indices: tuple[int, ...] = ()  # which steps are affected


# ─── Quality Report ───────────────────────────────────────────────────────────


@dataclass
class QualityReport:
    """
    Complete analysis result for one recorded trajectory.

    Produced by TrajectoryAnalyzer.analyze().
    Used to decide: use for training or send for human review.
    """

    session_id: str
    task: str
    total_steps: int
    action_counts: dict[str, int]  # {"click": 5, "type": 2}
    issues: list[TrajectoryIssue]  # everything detected
    quality_score: float  # 0.0 to 1.0
    use_for_training: bool  # True if score >= threshold
    summary: str  # one-line verdict

    @property
    def errors(self) -> list[TrajectoryIssue]:
        """Issues with ERROR severity."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[TrajectoryIssue]:
        """Issues with WARNING severity."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]


# ─── Analyzer ─────────────────────────────────────────────────────────────────


class TrajectoryAnalyzer:
    """
    Analyzes recorded trajectories and produces quality reports.

    Usage:
        analyzer = TrajectoryAnalyzer(quality_threshold=0.6)
        report   = analyzer.analyze(steps, session_id="abc", task="Open Gmail")

        if report.use_for_training:
            send_to_formatter(steps)
        else:
            send_to_human_review(steps, report)
            print(analyzer.format_report(report))
    """

    # Action types that appear at the end of every trajectory
    # They don't represent meaningful work — exclude from diversity checks
    _TERMINAL_TYPES = {ActionType.DONE.value, ActionType.FAIL.value}

    # Click-type actions — these have coordinates that we validate
    _CLICK_TYPES = {
        ActionType.CLICK.value,
        ActionType.DOUBLE_CLICK.value,
        ActionType.RIGHT_CLICK.value,
    }

    def __init__(
        self,
        quality_threshold: float = 0.6,
        loop_window: int = 4,
        max_same_action_ratio: float = 0.8,
    ) -> None:
        """
        Args:
            quality_threshold:     minimum score to approve for training
            loop_window:           maximum loop length to detect (2 to N)
            max_same_action_ratio: flag if one action type > this fraction
        """
        self._threshold = quality_threshold
        self._loop_window = loop_window
        self._max_same_ratio = max_same_action_ratio

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def analyze(
        self,
        steps: list[RecordedStep],
        session_id: str = "",
        task: str = "",
    ) -> QualityReport:
        """
        Run all quality checks and compute a final quality score.

        Args:
            steps:      trajectory steps in order
            session_id: for the report header
            task:       for the report header

        Returns:
            QualityReport with all findings
        """
        # ── Edge case: empty trajectory ───────────────────────────────────
        # An empty recording teaches the model nothing.
        # Return immediately with the worst possible score.
        if not steps:
            return QualityReport(
                session_id=session_id,
                task=task or "unknown",
                total_steps=0,
                action_counts={},
                issues=[
                    TrajectoryIssue(
                        severity=IssueSeverity.ERROR,
                        issue_type="empty_trajectory",
                        description="No steps recorded — nothing to train on",
                        step_indices=(),
                    )
                ],
                quality_score=0.0,
                use_for_training=False,
                summary="REJECTED — empty trajectory",
            )

        # ── Extract task from first step if not provided ──────────────────
        if not task:
            task = steps[0].task

        if not session_id:
            session_id = steps[0].session_id

        # ── Count action types ────────────────────────────────────────────
        action_counts = self.compute_action_statistics(steps)

        # ── Run all detectors ─────────────────────────────────────────────
        all_issues: list[TrajectoryIssue] = []
        all_issues.extend(self.detect_action_loops(steps))
        all_issues.extend(self.detect_redundant_actions(steps))
        all_issues.extend(self.detect_low_diversity(steps))
        all_issues.extend(self.detect_impossible_coordinates(steps))

        # ── Compute quality score ─────────────────────────────────────────
        score = 1.0
        for issue in all_issues:
            if issue.severity == IssueSeverity.ERROR:
                score -= 0.3
            elif issue.severity == IssueSeverity.WARNING:
                score -= 0.1

        # Clamp to [0.0, 1.0] — never negative, never above 1
        score = max(0.0, min(1.0, score))

        use_for_training = score >= self._threshold

        # ── Build summary sentence ────────────────────────────────────────
        error_count = sum(1 for i in all_issues if i.severity == IssueSeverity.ERROR)
        warning_count = sum(
            1 for i in all_issues if i.severity == IssueSeverity.WARNING
        )

        if use_for_training:
            summary = f"APPROVED (score={score:.2f}, {warning_count} warnings)"
        else:
            summary = (
                f"REJECTED (score={score:.2f}, "
                f"{error_count} errors, {warning_count} warnings)"
            )

        return QualityReport(
            session_id=session_id,
            task=task,
            total_steps=len(steps),
            action_counts=action_counts,
            issues=all_issues,
            quality_score=score,
            use_for_training=use_for_training,
            summary=summary,
        )

    # ── Detectors ─────────────────────────────────────────────────────────────

    def detect_action_loops(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect the agent stuck in a repeating sequence of actions.

        Algorithm:
          Try window sizes 2, 3, 4 (up to loop_window).
          For each window size W:
            Slide a window of size (2*W) across the steps.
            If the first W actions == the next W actions → loop found.

        Example (window=2):
          steps = [click, type, click, type, click, type, done]
          window at index 0: [click, type] == [click, type] → LOOP!
        """
        issues: list[TrajectoryIssue] = []

        # Extract just the action types for easy comparison
        action_types = [s.action.get("type", "") for s in steps]

        for window_size in range(2, self._loop_window + 1):
            # Need at least 2*window_size steps to have a full loop
            if len(action_types) < window_size * 2:
                continue

            # Slide window across all possible positions
            for start in range(len(action_types) - window_size * 2 + 1):
                first_half = action_types[start : start + window_size]
                second_half = action_types[
                    start + window_size : start + window_size * 2
                ]

                if first_half == second_half:
                    affected = tuple(range(start, start + window_size * 2))
                    issues.append(
                        TrajectoryIssue(
                            severity=IssueSeverity.WARNING,
                            issue_type="loop_detected",
                            description=(
                                f"Steps {start}–{start + window_size * 2 - 1} "
                                f"repeat sequence: {first_half}"
                            ),
                            step_indices=affected,
                        )
                    )
                    # Found a loop at this window size — stop sliding
                    # (avoid reporting the same loop multiple times)
                    break

        return issues

    def detect_redundant_actions(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect consecutive identical actions.

        Two actions are redundant if:
          - Same action type AND
          - Same coordinates (for click actions)
          - OR same text (for type actions)

        KEY exception: key actions are not flagged.
        Pressing Enter twice is often intentional (confirm two dialogs).

        Returns WARNING severity — redundant actions are suspicious
        but not always wrong.
        """
        issues: list[TrajectoryIssue] = []

        for i in range(len(steps) - 1):
            current = steps[i].action
            nxt = steps[i + 1].action

            current_type = current.get("type", "")
            next_type = nxt.get("type", "")

            # Different types → not redundant
            if current_type != next_type:
                continue

            # Key actions: intentionally skip (two enters is valid)
            if current_type == ActionType.KEY.value:
                continue

            # Click actions: redundant if same coordinates
            if current_type in self._CLICK_TYPES:
                current_coords = current.get("coordinates")
                next_coords = nxt.get("coordinates")
                if current_coords == next_coords and current_coords is not None:
                    issues.append(
                        TrajectoryIssue(
                            severity=IssueSeverity.WARNING,
                            issue_type="redundant_actions",
                            description=(
                                f"Steps {i} and {i+1}: consecutive {current_type} "
                                f"at same coordinates {current_coords}"
                            ),
                            step_indices=(i, i + 1),
                        )
                    )

            # Type actions: redundant if same text
            elif current_type == ActionType.TYPE.value:
                if current.get("text") == nxt.get("text"):
                    issues.append(
                        TrajectoryIssue(
                            severity=IssueSeverity.WARNING,
                            issue_type="redundant_actions",
                            description=(
                                f"Steps {i} and {i+1}: consecutive type "
                                f"with same text '{current.get('text', '')}'"
                            ),
                            step_indices=(i, i + 1),
                        )
                    )

            # Other action types (DONE/DONE, FAIL/FAIL): always redundant
            else:
                issues.append(
                    TrajectoryIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type="redundant_actions",
                        description=(
                            f"Steps {i} and {i+1}: consecutive identical "
                            f"{current_type} actions"
                        ),
                        step_indices=(i, i + 1),
                    )
                )

        return issues

    def detect_low_diversity(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect trajectories dominated by one action type.

        Why this matters for training:
          A trajectory with 90% CLICK actions teaches the model:
          "always click, rarely do anything else."
          We want diverse trajectories with mixed action types.

        Terminal actions (DONE, FAIL) are excluded from the ratio
        because they always appear exactly once at the end.
        """
        issues: list[TrajectoryIssue] = []

        # Only count non-terminal actions
        meaningful_types = [
            s.action.get("type", "")
            for s in steps
            if s.action.get("type", "") not in self._TERMINAL_TYPES
        ]

        if not meaningful_types:
            return issues

        # Count each type
        from collections import Counter

        counts = Counter(meaningful_types)
        total = len(meaningful_types)

        # Check if any single type dominates
        for action_type, count in counts.items():
            ratio = count / total
            if ratio > self._max_same_ratio:
                issues.append(
                    TrajectoryIssue(
                        severity=IssueSeverity.WARNING,
                        issue_type="low_diversity",
                        description=(
                            f"Action '{action_type}' appears {count}/{total} times "
                            f"({ratio*100:.0f}%) — model may overfit to this action"
                        ),
                        step_indices=(),
                    )
                )

        return issues

    def detect_impossible_coordinates(
        self,
        steps: list[RecordedStep],
    ) -> list[TrajectoryIssue]:
        """
        Detect click coordinates outside the valid [0, 1] range.

        Coordinates must be normalized between 0.0 and 1.0.
          0.0 = left/top edge of screen
          1.0 = right/bottom edge of screen

        Values outside this range cannot be converted to valid pixel
        positions and will cause the agent to crash or click off-screen.

        These are always ERROR severity — the step is unusable.
        """
        issues: list[TrajectoryIssue] = []

        for step in steps:
            action_type = step.action.get("type", "")

            if action_type not in self._CLICK_TYPES:
                continue

            coords = step.action.get("coordinates")
            if coords is None:
                continue

            # coords should be [x, y] where both are in [0.0, 1.0]
            try:
                x, y = float(coords[0]), float(coords[1])
            except (TypeError, ValueError, IndexError):
                issues.append(
                    TrajectoryIssue(
                        severity=IssueSeverity.ERROR,
                        issue_type="impossible_coordinates",
                        description=(
                            f"Step {step.step_index}: coordinates "
                            f"'{coords}' could not be parsed as [x, y]"
                        ),
                        step_indices=(step.step_index,),
                    )
                )
                continue

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(
                    TrajectoryIssue(
                        severity=IssueSeverity.ERROR,
                        issue_type="impossible_coordinates",
                        description=(
                            f"Step {step.step_index}: coordinates [{x:.3f}, {y:.3f}] "
                            f"are outside valid range [0.0, 1.0]"
                        ),
                        step_indices=(step.step_index,),
                    )
                )

        return issues

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_action_statistics(
        self,
        steps: list[RecordedStep],
    ) -> dict[str, int]:
        """
        Count how many times each action type appears.

        Returns dict sorted by count (highest first).
        Example: {"click": 8, "type": 3, "key": 2, "done": 1}
        """
        from collections import Counter

        counts = Counter(s.action.get("type", "unknown") for s in steps)
        # Sort by count descending, then alphabetically for ties
        return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

    # ── Report Formatting ─────────────────────────────────────────────────────

    def format_report(self, report: QualityReport) -> str:
        """
        Format a QualityReport as human-readable terminal output.

        Produces a clear, structured report showing:
          - Session info
          - Quality score
          - Action distribution with bar charts
          - All detected issues
          - Final verdict
        """
        SEP = "═" * 52
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append(SEP)
        lines.append(f"Quality Report — {report.session_id}")
        lines.append(f"Task: {report.task}")
        lines.append(SEP)

        # ── Score ─────────────────────────────────────────────────────────
        score_pct = report.quality_score * 100
        approved = "✓ use for training" if report.use_for_training else "✗ needs review"
        lines.append(f"Steps:         {report.total_steps}")
        lines.append(
            f"Quality Score: {report.quality_score:.2f} ({score_pct:.0f}%)  {approved}"
        )
        lines.append("")

        # ── Action Distribution ───────────────────────────────────────────
        if report.action_counts and report.total_steps > 0:
            lines.append("Action Distribution:")
            max_count = max(report.action_counts.values())
            bar_width = 20  # max bar length in characters

            for action_type, count in report.action_counts.items():
                pct = count / report.total_steps * 100
                bar_len = int((count / max_count) * bar_width)
                bar = "█" * bar_len
                lines.append(f"  {action_type:<14} {bar:<20} {count} ({pct:.0f}%)")

            lines.append("")

        # ── Issues ────────────────────────────────────────────────────────
        error_count = len(report.errors)
        warning_count = len(report.warnings)
        lines.append(
            f"Issues Found: {warning_count} warning(s), {error_count} error(s)"
        )

        if report.issues:
            for issue in report.issues:
                if issue.severity == IssueSeverity.ERROR:
                    icon = "✗ ERROR"
                elif issue.severity == IssueSeverity.WARNING:
                    icon = "⚠ WARNING"
                else:
                    icon = "ℹ INFO"

                lines.append(f"  {icon} [{issue.issue_type}]")
                lines.append(f"    {issue.description}")
        else:
            lines.append("  No issues detected.")

        lines.append("")

        # ── Verdict ───────────────────────────────────────────────────────
        if report.use_for_training:
            lines.append("Verdict: ✓ APPROVED for training")
        else:
            lines.append("Verdict: ✗ REJECTED — human review required")

        lines.append(SEP)

        return "\n".join(lines)
