"""Language detection and PDF text integrity verification service.

Detects English (en), Hindi (hi), Mixed (mixed), and Unknown (unknown).
Also validates extracted text against corruption, unmapped glyphs, and
legacy non-Unicode Indian font encodings (e.g. Kruti Dev, Chanakya, Shree Lipi).
"""

import enum
import re

from app.core.exceptions import CorruptedPDFTextError, ScannedPDFError


class DetectedLanguage(str, enum.Enum):
    ENGLISH = "en"
    HINDI = "hi"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# Known legacy non-Unicode Hindi / Indian fonts that map ASCII bytes to Devanagari glyphs
LEGACY_FONT_PATTERNS = [
    "kruti",
    "devlys",
    "chanakya",
    "walkman",
    "shree",
    "shree-dev",
    "sm7",
    "aps-",
    "shivaji",
    "4g-",
    "akruti",
    "dv-tt",
]

# Common legacy ASCII-substituted token patterns (e.g. "dksbZ", "iz/kkuea=h", "fo'ks\"k")
# where punctuation like ; = ' " / \ ~ ^ are embedded inside Latin letters
_LEGACY_PUNCT_WORD_RE = re.compile(r"\b[a-zA-Z]+[;=\'\"\\/~^]+[a-zA-Z]+\b")


class LanguageService:
    @staticmethod
    def detect_language(text: str) -> str:
        """Detect language of a text snippet or document: en, hi, mixed, or unknown."""
        if not text:
            return DetectedLanguage.UNKNOWN.value

        devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
        latin_count = len(re.findall(r"[a-zA-Z]", text))
        total_letters = devanagari_count + latin_count

        if total_letters < 20:
            return DetectedLanguage.UNKNOWN.value

        hi_ratio = devanagari_count / total_letters
        en_ratio = latin_count / total_letters

        if hi_ratio >= 0.70:
            return DetectedLanguage.HINDI.value
        elif en_ratio >= 0.85:
            return DetectedLanguage.ENGLISH.value
        elif hi_ratio >= 0.15 and en_ratio >= 0.15:
            return DetectedLanguage.MIXED.value
        elif hi_ratio > 0.40:
            return DetectedLanguage.HINDI.value
        else:
            return DetectedLanguage.UNKNOWN.value

    @staticmethod
    def validate_extracted_text(
        text: str, font_names: list[str] | None = None
    ) -> None:
        """Validate extracted text against empty content, Unicode corruption,
        and legacy non-Unicode Indian font encodings.

        Raises:
            ScannedPDFError: If text is empty or mostly whitespace.
            CorruptedPDFTextError: If text contains legacy font artifacts or corrupted Unicode.
        """
        if not text or not text.strip():
            raise ScannedPDFError("The PDF contains no extractable text.")

        cleaned = text.strip()
        length = len(cleaned)

        # 1. Check for excessive Unicode replacement characters (unmapped glyphs)
        replacement_count = cleaned.count("\ufffd")
        if length > 0 and (replacement_count / length) > 0.04:
            raise CorruptedPDFTextError(
                "The PDF extracted text contains excessive unmapped or corrupted "
                f"Unicode characters ({replacement_count} replacement characters)."
            )

        # 2. Check font names for known legacy non-Unicode font families
        devanagari_count = len(re.findall(r"[\u0900-\u097F]", cleaned))
        if font_names:
            lower_fonts = [f.lower() for f in font_names]
            for legacy in LEGACY_FONT_PATTERNS:
                if any(legacy in font for font in lower_fonts):
                    # If the document uses legacy Hindi fonts and has no real Devanagari Unicode,
                    # the extracted text is unreadable ASCII mapping.
                    if devanagari_count < 10:
                        raise CorruptedPDFTextError(
                            f"The PDF uses legacy non-Unicode Hindi font ({legacy}). "
                            "Extracted text is not valid Unicode Devanagari. "
                            "Please convert the PDF to Unicode or use a digital Unicode PDF."
                        )

        # 3. Heuristic for legacy ASCII Hindi encoding without font metadata
        # (e.g. repeated internal punctuation symbols inside alphabetic tokens)
        if devanagari_count < 10 and length >= 100:
            legacy_tokens = _LEGACY_PUNCT_WORD_RE.findall(cleaned)
            latin_words = re.findall(r"\b[a-zA-Z]{3,}\b", cleaned)
            if latin_words and len(legacy_tokens) >= 5 and (len(legacy_tokens) / len(latin_words)) > 0.15:
                raise CorruptedPDFTextError(
                    "The PDF appears to use legacy non-Unicode Hindi font encoding "
                    "with garbled ASCII character substitutions. Only Unicode-encoded PDFs are supported."
                )
