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
        """Render a PDF page to a pixmap image and run Tesseract OCR.
        
        Dynamically adjusts DPI based on physical page size so broadsheet newspapers
        do not explode into 38MP images that take hours to process.
        """
        try:
            rect = page.rect
            max_pt = max(rect.width, rect.height)
            if max_pt >= 1500:
                effective_dpi = min(self.dpi, 100)
            elif max_pt >= 1000:
                effective_dpi = min(self.dpi, 120)
            else:
                effective_dpi = min(self.dpi, 150)

            # Grayscale rendering avoids 3x RGB memory and speeds up binarization
            pix = page.get_pixmap(dpi=effective_dpi, colorspace=fitz.csGRAY)
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
        """Run OCR across all pages of a PDF document with progress tracking.

        Returns:
            List of (page_number, text) tuples.
        """
        results: list[tuple[int, str]] = []
        total_pages = len(document)
        for page_number, page in enumerate(document, start=1):
            logger.info("Running OCR on page %d/%d...", page_number, total_pages)
            page_text = self.extract_text_from_page(page)
            if page_text:
                results.append((page_number, page_text))
                logger.info(
                    "OCR completed page %d/%d (%d chars)",
                    page_number,
                    total_pages,
                    len(page_text),
                )
            else:
                logger.warning("OCR extracted 0 chars on page %d/%d", page_number, total_pages)
        return results
