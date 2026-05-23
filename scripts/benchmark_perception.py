"""
Benchmark: Sequential vs Parallel perception.

Measures real time difference between:
  - Old approach: capture then get_ui_tree (sequential)
  - New approach: asyncio.gather (parallel)

Run with:
    python scripts/benchmark_perception.py
"""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from visionnav.perception.fusion import fuse_parallel
from visionnav.perception.ocr import TextRegion


class MockPlatform:
    """
    Simulates a real platform adapter.

    We intentionally add asyncio.sleep delays to simulate
    real I/O latency from:
      - screenshot capture
      - accessibility API calls
    """

    async def capture(self) -> tuple[np.ndarray, dict]:
        # Simulate screenshot capture latency (~50ms)
        await asyncio.sleep(0.05)

        image = np.zeros((720, 1280, 3), dtype=np.uint8)

        meta = {
            "width": 1280,
            "height": 720,
            "platform": "desktop",
            "path": "mock.png",
        }

        return image, meta

    async def get_ui_tree(self) -> list[dict]:
        # Simulate accessibility/UI tree extraction latency (~30ms)
        await asyncio.sleep(0.03)

        return [
            {
                "type": "button",
                "label": "Submit",
                "bounds": [100, 100, 200, 150],
            }
        ]


class MockOCREngine:
    """
    Simulates synchronous OCR work.

    OCR is intentionally synchronous because it represents
    CPU-bound processing rather than async I/O.
    """

    def run(self, image: np.ndarray) -> list[TextRegion]:
        # Simulate OCR compute time (~15ms)
        time.sleep(0.015)

        return [
            TextRegion(
                text="Hello World",
                bbox=(0.1, 0.1, 0.3, 0.2),
                confidence=0.95,
            )
        ]


async def sequential_perception(
    platform: MockPlatform,
    ocr_engine: MockOCREngine,
) -> None:
    """
    Old sequential approach.

    Total expected latency:
        capture (50ms)
      + ui tree (30ms)
      + OCR (15ms)
      = ~95ms
    """

    image, meta = await platform.capture()

    ui_tree = await platform.get_ui_tree()

    ocr_engine.run(image)

    _ = (meta, ui_tree)


async def parallel_perception(
    platform: MockPlatform,
    ocr_engine: MockOCREngine,
) -> None:
    """
    New parallel approach using asyncio.gather.

    Total expected latency:
        max(capture, ui_tree)
      + OCR
      = ~65ms
    """

    await fuse_parallel(
        platform=platform,
        ocr_engine=ocr_engine,
    )


async def benchmark(
    runs: int = 10,
) -> None:
    platform = MockPlatform()
    ocr_engine = MockOCREngine()

    sequential_times: list[float] = []
    parallel_times: list[float] = []

    # Benchmark sequential approach.
    for _ in range(runs):
        start = time.perf_counter()

        await sequential_perception(platform, ocr_engine)

        end = time.perf_counter()

        sequential_times.append((end - start) * 1000)

    # Benchmark parallel approach.
    for _ in range(runs):
        start = time.perf_counter()

        await parallel_perception(platform, ocr_engine)

        end = time.perf_counter()

        parallel_times.append((end - start) * 1000)

    sequential_avg = sum(sequential_times) / len(sequential_times)
    parallel_avg = sum(parallel_times) / len(parallel_times)

    improvement = ((sequential_avg - parallel_avg) / sequential_avg) * 100

    print()
    print(f"Perception Benchmark ({runs} runs each)")
    print("════════════════════════════════════════")

    print(f"Sequential:  {sequential_avg:.1f} ms average")
    print(f"Parallel:    {parallel_avg:.1f} ms average")
    print(f"Improvement: {improvement:.1f}% faster")

    print()
    print("Step breakdown:")
    print("  Capture:  50.0 ms")
    print("  UI Tree:  30.0 ms")
    print("  OCR:      15.0 ms")
    print()


if __name__ == "__main__":
    asyncio.run(benchmark())
