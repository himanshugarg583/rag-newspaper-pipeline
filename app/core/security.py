"""Security helpers: filename sanitization and upload path containment.

Authentication / authorization are intentionally not implemented yet; this
module is the extension point for adding them later.
"""

import re
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from a client filename."""
    name = Path(filename).name  # drop any directory components
    name = _FILENAME_SAFE_RE.sub("_", name).strip("._")
    return name or "upload.pdf"


def generate_stored_filename(original_filename: str) -> str:
    """Generate a unique, safe filename for storage on disk."""
    safe = sanitize_filename(original_filename)
    suffix = Path(safe).suffix.lower() or ".pdf"
    return f"{uuid.uuid4().hex}{suffix}"


def resolve_upload_path(upload_dir: str, stored_filename: str) -> Path:
    """Resolve a storage path and ensure it stays inside the upload directory."""
    base = Path(upload_dir).resolve()
    target = (base / stored_filename).resolve()
    if base != target and base not in target.parents:
        raise ValueError("Resolved file path escapes the upload directory.")
    return target
