"""
Centralized logging engine for AI Voice Studio.

Provides standardized logging formatters and console/file handlers.
"""

import logging
import os
import re
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Lines written every time a client polls a job's status or the health endpoint.
_POLL_MESSAGE_PATTERNS = (
    re.compile(r"^Calling GET /v1/jobs/[^/\s]+ endpoint$"),
    re.compile(r"^Executing JobController\.get_job_status for job '"),
    re.compile(r"^Executing JobCRUD\.get_job_by_id for job '"),
    re.compile(r"^Calling GET (?:/v1)?/health endpoint$"),
    re.compile(r"^Executing CatalogController\.get_health_status$"),
)
# Uvicorn access lines for the same polls: 127.0.0.1:5000 - "GET /v1/jobs/abc HTTP/1.1" 200
# (uvicorn adds the " OK" phrase only when it formats the line, so it is optional here).
_POLL_ACCESS_PATTERN = re.compile(r'^\S+ - ("GET (?:/v1/jobs/[^\s/?]+|/v1/health|/health) HTTP/[\d.]+" 200)(?: OK)?$')


class PollingNoiseFilter(logging.Filter):
    """
    Show a repeated polling log line once per time window instead of once per request.

    A browser polling a running job asks every second, which buried the real events in
    thousands of identical lines. Only successful (HTTP 200) polling lines of INFO level or
    lower are throttled; warnings, errors and every other message always pass. Set
    LOG_POLL_THROTTLE_SECONDS=0 to see every line again.
    """

    _MAX_TRACKED = 2048

    def __init__(self, window_seconds: float | None = None) -> None:
        """
        Configure the throttle window.

        Args:
            window_seconds (float | None): Seconds a repeated line stays hidden; None reads
                LOG_POLL_THROTTLE_SECONDS (default 30).

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        if window_seconds is None:
            try:
                window_seconds = float(os.environ.get("LOG_POLL_THROTTLE_SECONDS", "30"))
            except ValueError:
                window_seconds = 30.0
        self.window_seconds = window_seconds
        self._last_shown: Dict[str, float] = {}
        self._lock = threading.Lock()

    def _polling_key(self, record: logging.LogRecord) -> str | None:
        """
        Identify whether a record is a polling line and return its throttle key.

        Args:
            record (logging.LogRecord): The record being logged.

        Returns:
            str | None: Key for a polling line, or None for any other record.

        Raises:
            None
        """
        message = record.getMessage()
        if record.name == "uvicorn.access":
            match = _POLL_ACCESS_PATTERN.match(message)
            return f"access|{match.group(1)}" if match else None
        if any(pattern.match(message) for pattern in _POLL_MESSAGE_PATTERNS):
            return f"app|{message}"
        return None

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Decide whether a record is written.

        Args:
            record (logging.LogRecord): The record being logged.

        Returns:
            bool: False only for a polling line already shown within the window.

        Raises:
            None
        """
        if self.window_seconds <= 0 or record.levelno > logging.INFO:
            return True
        key = self._polling_key(record)
        if key is None:
            return True
        now = time.monotonic()
        with self._lock:
            last = self._last_shown.get(key)
            if last is not None and now - last < self.window_seconds:
                return False
            if len(self._last_shown) >= self._MAX_TRACKED:
                cutoff = now - self.window_seconds
                self._last_shown = {k: t for k, t in self._last_shown.items() if t >= cutoff}
            self._last_shown[key] = now
        return True


# One shared instance so every logger throttles against the same record of what was shown.
POLLING_FILTER = PollingNoiseFilter()


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

        # Filter on the logger (not each handler) so a line is judged once, then goes to console and file.
        log_instance.addFilter(POLLING_FILTER)
        log_instance.propagate = False

    return log_instance
