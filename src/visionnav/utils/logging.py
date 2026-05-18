"""Structured logging setup."""
from __future__ import annotations
import logging, os
import structlog


def setup_logging(level: str | None = None) -> None:
    lvl     = level or os.getenv("LOG_LEVEL", "INFO")
    is_dev  = os.getenv("VISIONNAV_ENV", "development") == "development"
    logging.basicConfig(format="%(message)s",
                        level=getattr(logging, lvl.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if is_dev else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, lvl.upper(), logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str = "") -> structlog.BoundLogger:
    return structlog.get_logger(name)
