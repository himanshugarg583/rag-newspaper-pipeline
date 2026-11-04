"""PDF extraction service using PyMuPDF and Tesseract OCR.

Extracts digital/searchable text directly and falls back to Tesseract OCR
(English + Hindi) for scanned, image-based, or epaper PDFs.
"""

from dataclasses import dataclass

import fitz  # PyMuPDF

from app.core.config import settings
from app.core.exceptions import InvalidPDFError, ScannedPDFError
from app.core.logging import get_logger
from app.services.language_service import LanguageService
from app.services.ocr_service import OCRService

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractionResult:
    total_pages: int
    pages: list[ExtractedPage]
    font_names: list[str]


class PDFService:
    """Extract text page by page, preserving original PDF page numbers and Unicode scripts,
    falling back to OCR when necessary.
    """

    def __init__(self, ocr_service: OCRService | None = None):
        self.ocr_service = ocr_service or OCRService()

    def extract_pages(self, pdf_path: str) -> ExtractionResult:
        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            raise InvalidPDFError(f"Unable to open PDF: {exc}") from exc

        try:
            if document.is_encrypted and not document.authenticate(""):
                raise InvalidPDFError("Encrypted PDFs are not supported.")

            total_pages = document.page_count
            if total_pages == 0:
                raise InvalidPDFError("PDF contains no pages.")

            pages: list[ExtractedPage] = []
            font_names: set[str] = set()

            for page_number, page in enumerate(document, start=1):
                try:
                    for f in page.get_fonts():
                        if len(f) > 3 and f[3]:
                            font_names.add(str(f[3]))
                except Exception:
                    pass

                text = page.get_text("text")
                if text and text.strip():
                    pages.append(
                        ExtractedPage(page_number=page_number, text=text.strip())
                    )

            total_chars = sum(len(p.text) for p in pages)

            # If digital extraction yielded insufficient text, trigger OCR fallback
            if total_chars < settings.min_extractable_chars and settings.enable_ocr:
                logger.info(
                    "Digital extraction returned insufficient text (%d chars). Running OCR on %s...",
                    total_chars,
                    pdf_path,
                )
                try:
                    ocr_results = self.ocr_service.extract_text_from_document(document)
                    if ocr_results:
                        pages = [
                            ExtractedPage(page_number=p_num, text=p_text)
                            for p_num, p_text in ocr_results
                        ]
                        total_chars = sum(len(p.text) for p in pages)
                except Exception as ocr_err:
                    logger.warning("OCR processing encountered an issue: %s", ocr_err)

            if not pages or total_chars < 10:
                raise ScannedPDFError(
                    "The PDF appears to be blank, empty, or unreadable even after OCR processing."
                )

            # Validate extracted text for legacy non-Unicode fonts or corruption
            combined_text = "\n".join(p.text for p in pages)
            LanguageService.validate_extracted_text(combined_text, list(font_names))

            logger.info(
                "Extracted %d pages (%d chars) from %s",
                len(pages),
                total_chars,
                pdf_path,
            )
            return ExtractionResult(
                total_pages=total_pages,
                pages=pages,
                font_names=list(font_names),
            )
        finally:
            document.close()
