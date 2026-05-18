"""Retry decorator with exponential backoff."""
from __future__ import annotations
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)


def with_retry(max_attempts: int = 3, min_wait: float = 1.0,
               max_wait: float = 8.0, exceptions: tuple = (Exception,)):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
