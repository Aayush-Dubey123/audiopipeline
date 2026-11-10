"""
Centralized logging engine for AI Voice Studio.

Provides standardized logging formatters and console/file handlers.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def logger(name: str = __name__) -> logging.Logger:
    """
    Obtain a configured logger instance with standard formatting.

    Initializes stream and file handlers if not already configured for the
    requested logger module namespace.

    Args:
        name (str): The namespace name for the logger, typically __name__.

    Returns:
        logging.Logger: Configured standard Logger instance.

    Raises:
        None
    """
    log_instance = logging.getLogger(name)

    if not log_instance.handlers:
        log_instance.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        log_instance.addHandler(console_handler)

        # File handler (if logs directory exists or can be created)
        try:
            # Anchored to backend/logs and size-capped so logs never pile up
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                _LOG_DIR / "app.log",
                maxBytes=int(os.environ.get("LOG_MAX_BYTES", 1_000_000)),
                backupCount=int(os.environ.get("LOG_BACKUP_COUNT", 1)),
                encoding="utf-8",
                delay=True,
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            log_instance.addHandler(file_handler)
        except Exception:
            # Fallback to console only if file handler cannot be established
            pass

        log_instance.propagate = False

    return log_instance
