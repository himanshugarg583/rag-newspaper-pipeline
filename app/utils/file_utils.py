"""Filesystem utilities for stored newspaper PDFs."""

import os
from pathlib import Path

from app.core.logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_bytes(path: Path, data: bytes) -> None:
    ensure_directory(str(path.parent))
    with open(path, "wb") as handle:
        handle.write(data)


def delete_file_safely(path: str) -> bool:
    """Delete a file; missing files are handled gracefully."""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        logger.warning("File already missing during deletion: %s", path)
        return False
    except OSError:
        logger.exception("Failed to delete file: %s", path)
        return False
