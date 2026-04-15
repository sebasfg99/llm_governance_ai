# ============================================================
# app/core/logging.py — Structured application logging
#
# WHY THIS FILE EXISTS:
# Instead of using print() everywhere, we use Python's built-in
# logging module with a consistent format across all modules.
#
# WHY STRUCTURED LOGGING:
# In production, logs are often collected by tools like
# Datadog, CloudWatch, or Grafana Loki. These tools parse
# JSON logs much more reliably than plain text. Structured
# logging lets you filter/search by fields like tenant_id,
# trace_id, or level.
#
# HOW TO USE IN OTHER FILES:
#   from app.core.logging import get_logger
#   logger = get_logger(__name__)
#   logger.info("Document processed", extra={"doc_id": "abc123"})
#
# TODO: Switch to structlog library for richer structured logging
# TODO: Add log correlation IDs (trace_id per request)
# ============================================================

import logging
import sys
from app.core.config import settings


def configure_logging() -> None:
    """
    Set up the root logger with the correct log level and format.

    This is called once at application startup. All loggers
    created with get_logger() will inherit this configuration.
    """
    # Parse the log level string from settings (e.g., "INFO" → logging.INFO)
    # If the string is invalid, default to INFO to avoid crashing at startup
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure the format of log messages
    # %(asctime)s    — timestamp
    # %(name)s       — logger name (usually the module name)
    # %(levelname)s  — DEBUG, INFO, WARNING, ERROR, CRITICAL
    # %(message)s    — the actual log message
    log_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    # Configure the root logger
    # All loggers created via logging.getLogger() are children of the root logger
    # and will inherit this level and handler unless overridden
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            # Log to stdout (not stderr) so Docker and cloud platforms capture it
            logging.StreamHandler(sys.stdout)
        ],
    )

    # Quiet down noisy third-party libraries
    # These libraries log a lot at DEBUG level, which clutters our logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("supabase").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Factory function to get a named logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Something happened")
        logger.error("Something failed", exc_info=True)  # includes stack trace

    Args:
        name: Typically __name__ (the module's full dotted path)
              e.g., "app.domain.documents.service"

    Returns:
        A configured Logger instance
    """
    # Ensure logging is configured before returning a logger
    # configure_logging() is idempotent — safe to call multiple times
    configure_logging()

    return logging.getLogger(name)


# ============================================================
# Run configuration once at import time
# This ensures logging is ready as soon as any module is imported
# ============================================================
configure_logging()
