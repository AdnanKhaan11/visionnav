import pytest
from visionnav.actions.recorder import RecordedStep
from visionnav.actions.trajectory_analyzer import (
    TrajectoryAnalyzer,
    IssueSeverity,
    QualityReport,
)
from visionnav.actions.schema import ActionType
from datetime import datetime, timezone


def make_step(
    index: int,
    action_type: ActionType = ActionType.CLICK,
    coordinates: tuple = (0.5, 0.5),
) -> RecordedStep:
    action_dict = {"type": action_type.value}
    if action_type in (ActionType.CLICK, ActionType.DOUBLE_CLICK):
        action_dict["coordinates"] = list(coordinates)
    return RecordedStep(
        step_index=index,
        task="test task",
        action=action_dict,
        screenshot_path=f"step_{index:03d}.png",
        ocr_text="some text",
        screen_width=1920,
        screen_height=1080,
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id="test-session",
    )


# Test 1: Empty trajectory is flagged as error
def test_empty_trajectory_is_error():
    analyzer = TrajectoryAnalyzer()
    report = analyzer.analyze([])
    assert report.quality_score == 0.0
    assert report.use_for_training is False
    assert any(i.severity == IssueSeverity.ERROR for i in report.issues)


# Test 2: Clean trajectory gets high score
def test_clean_trajectory_high_score():
    steps = [
        make_step(0, ActionType.KEY),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.CLICK),
        make_step(3, ActionType.DONE),
    ]
    report = TrajectoryAnalyzer().analyze(steps)
    assert report.quality_score >= 0.8
    assert report.use_for_training is True


# Test 3: Loop detection works
def test_loop_detection():
    # Pattern: click, type, click, type → loop of length 2
    steps = [
        make_step(0, ActionType.CLICK),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.CLICK),
        make_step(3, ActionType.TYPE),
        make_step(4, ActionType.DONE),
    ]
    analyzer = TrajectoryAnalyzer()
    issues = analyzer.detect_action_loops(steps)
    assert len(issues) > 0
    assert any("loop" in i.issue_type.lower() for i in issues)


# Test 4: No false positive on non-looping trajectory
def test_no_loop_false_positive():
    steps = [
        make_step(0, ActionType.KEY),
        make_step(1, ActionType.TYPE),
        make_step(2, ActionType.SCROLL),
        make_step(3, ActionType.CLICK),
        make_step(4, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_action_loops(steps)
    assert len(issues) == 0


# Test 5: Redundant actions detected
def test_redundant_actions_detected():
    # Two consecutive clicks at same coordinates
    steps = [
        make_step(0, ActionType.CLICK, (0.5, 0.5)),
        make_step(1, ActionType.CLICK, (0.5, 0.5)),
        make_step(2, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_redundant_actions(steps)
    assert len(issues) > 0


# Test 6: Low diversity flagged
def test_low_diversity_flagged():
    # 9 clicks and 1 done = 100% click ratio
    steps = [make_step(i, ActionType.CLICK) for i in range(9)]
    steps.append(make_step(9, ActionType.DONE))
    issues = TrajectoryAnalyzer().detect_low_diversity(steps)
    assert len(issues) > 0


# Test 7: Impossible coordinates flagged as error
def test_impossible_coordinates_flagged():
    steps = [
        make_step(0, ActionType.CLICK, (1.5, 0.5)),  # x > 1.0 ← impossible
        make_step(1, ActionType.DONE),
    ]
    issues = TrajectoryAnalyzer().detect_impossible_coordinates(steps)
    assert len(issues) > 0
    assert all(i.severity == IssueSeverity.ERROR for i in issues)


# Test 8: quality_score decreases with issues
def test_quality_score_decreases_with_issues():
    clean = [
        make_step(i, t)
        for i, t in enumerate(
            [ActionType.KEY, ActionType.TYPE, ActionType.CLICK, ActionType.DONE]
        )
    ]
    looped = [
        make_step(i, t)
        for i, t in enumerate(
            [
                ActionType.CLICK,
                ActionType.TYPE,
                ActionType.CLICK,
                ActionType.TYPE,
                ActionType.DONE,
            ]
        )
    ]

    clean_score = TrajectoryAnalyzer().analyze(clean).quality_score
    looped_score = TrajectoryAnalyzer().analyze(looped).quality_score
    assert clean_score > looped_score


# Test 9: format_report returns non-empty string
def test_format_report_returns_string():
    steps = [make_step(0, ActionType.CLICK), make_step(1, ActionType.DONE)]
    report = TrajectoryAnalyzer().analyze(steps)
    text = TrajectoryAnalyzer().format_report(report)
    assert isinstance(text, str)
    assert len(text) > 50
    assert "Quality" in text
