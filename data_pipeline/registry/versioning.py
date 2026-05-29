"""
Dataset version management — semantic versioning for datasets.

Dataset versioning is stricter than software versioning:
  MAJOR.MINOR.PATCH

  PATCH: fixed bugs in existing samples (wrong coordinates, OCR errors)
  MINOR: added new samples, new languages, new task categories
  MAJOR: changed schema (new required fields, removed fields, type changes)

A model trained on v2.1.0 must produce reproducible results.
This means v2.1.0 must always refer to exactly the same set of samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


@dataclass(frozen=True)
class DatasetVersion:
    """
    Immutable semantic version for a dataset release.
    frozen=True: versions are facts, never changed.
    """

    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "DatasetVersion") -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )

    def __le__(self, other: "DatasetVersion") -> bool:
        return self == other or self < other

    def bump_patch(self) -> "DatasetVersion":
        """Bug fixes in existing samples."""
        return DatasetVersion(self.major, self.minor, self.patch + 1)

    def bump_minor(self) -> "DatasetVersion":
        """New samples added, backward compatible."""
        return DatasetVersion(self.major, self.minor + 1, 0)

    def bump_major(self) -> "DatasetVersion":
        """Schema change — requires migration."""
        return DatasetVersion(self.major + 1, 0, 0)

    @classmethod
    def parse(cls, version_str: str) -> "DatasetVersion":
        """
        Parse a version string into a DatasetVersion.

        Raises:
            ValueError: if string is not valid semver
        """
        match = _VERSION_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(
                f"Invalid version string: {version_str!r}. "
                f"Expected format: MAJOR.MINOR.PATCH (e.g. '1.2.0')"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )

    @classmethod
    def initial(cls) -> "DatasetVersion":
        """First version of any dataset."""
        return cls(1, 0, 0)
