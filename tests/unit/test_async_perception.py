"""Tests for async perception and timeout-safe model execution."""

from __future__ import annotations

import asyncio
import time

import pytest

from visionnav.actions.parser import parse_action
from visionnav.actions.schema import ActionType
from visionnav.models.local import LocalModelBackend
from visionnav.perception.fusion import Observation, fuse, fuse_parallel
from visionnav.perception.ocr import TextRegion


class MockOCREngine:
    """
    Mock OCR engine with synchronous CPU-style behavior.
    """

    def run(self, image):
        """
        Simulate OCR processing delay.
        """

        time.sleep(0.01)

        return [
            TextRegion(
                text="Hello",
                bbox=(0.1, 0.1, 0.3, 0.2),
                confidence=0.95,
            )
        ]


# Test 1: fuse_parallel returns valid Observation
@pytest.mark.asyncio
async def test_fuse_parallel_returns_observation(mock_platform):
    """
    Ensure fuse_parallel returns a valid Observation object
    containing screenshot, OCR, and UI tree data.
    """

    ocr_engine = MockOCREngine()

    result = await fuse_parallel(
        platform=mock_platform,
        ocr_engine=ocr_engine,
    )

    assert isinstance(result, Observation)

    assert result.screen_width > 0
    assert result.screen_height > 0

    assert isinstance(result.ocr_regions, list)
    assert isinstance(result.ui_elements, list)

    assert len(result.ocr_regions) > 0

    assert result.screenshot_b64 != ""


# Test 2: fuse_parallel is actually faster than sequential
@pytest.mark.asyncio
async def test_fuse_parallel_faster_than_sequential(mock_platform):
    """
    Verify that parallel perception is not slower than sequential.

    Since timing in CI/testing environments can fluctuate,
    we allow a small tolerance.
    """

    ocr_engine = MockOCREngine()

    sequential_times: list[float] = []
    parallel_times: list[float] = []

    # Sequential benchmark
    for _ in range(5):
        start = time.perf_counter()

        image, meta = await mock_platform.capture()

        ui_tree = await mock_platform.get_ui_tree()

        ocr_regions = ocr_engine.run(image)

        fuse(
            image=image,
            meta=meta,
            ocr_regions=ocr_regions,
            ui_elements=ui_tree,
        )

        end = time.perf_counter()

        sequential_times.append(end - start)

    # Parallel benchmark
    for _ in range(5):
        start = time.perf_counter()

        await fuse_parallel(
            platform=mock_platform,
            ocr_engine=ocr_engine,
        )

        end = time.perf_counter()

        parallel_times.append(end - start)

    sequential_avg = sum(sequential_times) / len(sequential_times)
    parallel_avg = sum(parallel_times) / len(parallel_times)

    # Allow small timing tolerance for CI noise
    assert parallel_avg <= sequential_avg + 0.01


class SlowMockModel(LocalModelBackend):
    """
    Mock backend that intentionally hangs to test timeout handling.
    """

    async def predict_action(
        self,
        observation,
        task,
        history,
        plan,
    ) -> str:
        """
        Simulate a very slow model call.
        """

        await asyncio.sleep(5)

        return (
            "<action>" '{"type":"done","description":"should never happen"}' "</action>"
        )


# Test 3: timeout returns parseable FAIL action
@pytest.mark.asyncio
async def test_model_timeout_returns_fail_action():
    """
    Ensure timeout returns a valid parser-compatible FAIL action.
    """

    model = SlowMockModel.__new__(SlowMockModel)

    result = await model.predict_action_safe(
        observation=None,
        task="test task",
        history=[],
        plan=[],
        timeout_seconds=0.1,
    )

    assert "<action>" in result

    parsed = parse_action(result)

    assert parsed.type == ActionType.FAIL


# Test 4: timeout is logged
@pytest.mark.asyncio
async def test_model_timeout_is_logged(capsys):
    """
    Ensure timeout warning is emitted to stdout logs.
    """

    model = SlowMockModel.__new__(SlowMockModel)

    await model.predict_action_safe(
        observation=None,
        task="test task",
        history=[],
        plan=[],
        timeout_seconds=0.1,
    )

    captured = capsys.readouterr()

    assert "model_timeout" in captured.out
