"""OCR service for extracting text from scanned or image-based newspaper PDFs.

Uses Tesseract OCR via pytesseract with multi-language support (English + Hindi: eng+hin).
"""

import io

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OCRService:
    def __init__(
        self,
        languages: str | None = None,
        dpi: int | None = None,
    ):
        self.languages = languages or settings.ocr_languages
        self.dpi = dpi or settings.ocr_dpi

    def extract_text_from_page(self, page: fitz.Page) -> str:
        """Render a PDF page to a pixmap image and run Tesseract OCR."""
        try:
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang=self.languages)
            return text.strip() if text else ""
        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract-OCR executable not found on the system.")
            raise
        except Exception as exc:
            logger.warning("OCR failed on page %d: %s", page.number + 1, exc)
            return ""

    def extract_text_from_document(
        self, document: fitz.Document
    ) -> list[tuple[int, str]]:
        """Run OCR across all pages of a PDF document.

        Returns:
            List of (page_number, text) tuples.
        """
        results: list[tuple[int, str]] = []
        for page_number, page in enumerate(document, start=1):
            page_text = self.extract_text_from_page(page)
            if page_text:
                results.append((page_number, page_text))
        return results
