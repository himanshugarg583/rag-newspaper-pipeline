"""Structured logging configuration."""

import json
import logging
import sys
from datetime import datetime, timezone

from app.core.config import settings

# Keys that must never appear in log output.
SENSITIVE_KEYS = {"openai_api_key", "api_key", "password", "database_url", "secret"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            payload.update(
                {k: v for k, v in extra.items() if k.lower() not in SENSITIVE_KEYS}
            )
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    """Configure root logging once at application startup."""
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    # Avoid duplicated handlers when reloading.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if settings.app_env == "development" and settings.debug:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        )
    else:
        handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
