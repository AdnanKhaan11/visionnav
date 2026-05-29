"""
Raw JSONL reader — streams raw recording files line by line.

Design principle: the reader does NOTHING except read.
No validation, no transformation, no business logic.
It reads bytes and yields dicts.
Every other concern belongs in a different module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import structlog

from data_pipeline.core.exceptions import IngestionError

log = structlog.get_logger(__name__)


@dataclass
class RawLine:
    """
    One raw line from a JSONL recording file.
    Preserves exact provenance: which file, which line.
    This information feeds into SampleLineage later.
    """

    data: dict
    source_file: str
    line_number: int


def stream_jsonl(path: Path) -> Iterator[RawLine]:
    """
    Stream lines from a JSONL file one at a time.
    Never loads entire file into memory.
    Skips and logs corrupt lines — does not crash.

    Args:
        path: path to .jsonl file

    Yields:
        RawLine with parsed dict + provenance

    Raises:
        IngestionError: if file cannot be opened at all
    """
    path = Path(path)

    if not path.exists():
        raise IngestionError(f"Recording file not found: {path}")

    if not path.suffix == ".jsonl":
        log.warning("unexpected_extension", path=str(path), expected=".jsonl")

    line_count = 0
    parsed_count = 0
    error_count = 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line_count += 1
                stripped = raw_line.strip()

                if not stripped:
                    continue  # skip empty lines (common at file end)

                try:
                    data = json.loads(stripped)
                    parsed_count += 1
                    yield RawLine(
                        data=data,
                        source_file=str(path),
                        line_number=line_count,
                    )
                except json.JSONDecodeError as exc:
                    error_count += 1
                    log.warning(
                        "jsonl_parse_error",
                        file=str(path),
                        line=line_count,
                        error=str(exc),
                        preview=stripped[:80],
                    )

    except OSError as exc:
        raise IngestionError(f"Cannot read file {path}: {exc}")

    log.info(
        "jsonl_stream_complete",
        file=str(path),
        total_lines=line_count,
        parsed=parsed_count,
        errors=error_count,
    )


def stream_directory(
    directory: Path,
    pattern: str = "*.jsonl",
    recursive: bool = False,
) -> Iterator[tuple[Path, RawLine]]:
    """
    Stream all JSONL files in a directory.
    Yields (source_path, RawLine) pairs.

    Args:
        directory: directory to scan
        pattern:   glob pattern for files
        recursive: if True, scan subdirectories too

    Yields:
        (file_path, RawLine) tuples
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise IngestionError(f"Not a directory: {directory}")

    glob_fn = directory.rglob if recursive else directory.glob
    files = sorted(glob_fn(pattern))

    if not files:
        log.warning("no_files_found", directory=str(directory), pattern=pattern)
        return

    log.info("scanning_directory", directory=str(directory), files_found=len(files))

    for file_path in files:
        try:
            for raw_line in stream_jsonl(file_path):
                yield file_path, raw_line
        except IngestionError as exc:
            log.error("file_ingestion_failed", file=str(file_path), error=str(exc))
            continue  # skip bad file, continue with others
