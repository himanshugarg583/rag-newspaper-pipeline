"""Text cleaning for raw PDF extraction output.

Removes common PDF extraction artifacts without destroying paragraph
boundaries or meaningful newspaper content.
"""

import re
import unicodedata


class TextCleaningService:
    # 3+ consecutive newlines -> paragraph break
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    # Runs of spaces/tabs -> single space
    _MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
    # Hyphenation at line end: "devel-\nopment" -> "development"
    _HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
    # Control characters except newline/tab
    _CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Normalize unicode forms (ligatures, odd quote variants, etc.).
        cleaned = unicodedata.normalize("NFKC", text)

        # Normalize line endings and strip control characters.
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self._CONTROL_RE.sub("", cleaned)

        # Repair words hyphenated across line breaks.
        cleaned = self._HYPHEN_BREAK_RE.sub(r"\1\2", cleaned)

        # Collapse whitespace while preserving paragraph breaks.
        cleaned = self._MULTI_SPACE_RE.sub(" ", cleaned)
        cleaned = self._MULTI_NEWLINE_RE.sub("\n\n", cleaned)

        # Trim trailing spaces per line.
        cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))

        return cleaned.strip()
