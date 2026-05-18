"""OS accessibility tree extraction."""
from __future__ import annotations
import platform
import structlog

log = structlog.get_logger(__name__)


def get_ui_tree() -> list[dict]:
    """Return UI element list. Returns [] gracefully if unavailable."""
    system = platform.system()
    try:
        if system == "Windows":  return _windows()
        if system == "Darwin":   return _macos()
        if system == "Linux":    return _linux()
        return []
    except Exception as exc:
        log.warning("ui_tree_failed", system=system, error=str(exc))
        return []


def _windows() -> list[dict]:
    # TODO Phase 5: pywinauto / comtypes UIAutomation
    return []

def _macos() -> list[dict]:
    # TODO Phase 5: ApplicationServices / Quartz Accessibility API
    return []

def _linux() -> list[dict]:
    # TODO Phase 5: pyatspi AT-SPI
    return []
