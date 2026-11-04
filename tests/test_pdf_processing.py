"""PDF extraction and text cleaning tests."""

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import InvalidPDFError, ScannedPDFError
from app.services.pdf_service import PDFService
from app.services.text_cleaning_service import TextCleaningService


class TestPDFExtraction:
    def test_digital_pdf_extraction(self, tmp_path, digital_pdf_bytes):
        pdf_path = tmp_path / "digital.pdf"
        pdf_path.write_bytes(digital_pdf_bytes)

        result = PDFService().extract_pages(str(pdf_path))

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[1].page_number == 2
        assert "artificial intelligence" in result.pages[0].text.lower()

    def test_scanned_pdf_with_ocr_fallback(self, tmp_path, empty_pdf_bytes):
        """Test that scanned PDFs automatically trigger OCR and extract text."""
        pdf_path = tmp_path / "scanned_news.pdf"
        pdf_path.write_bytes(empty_pdf_bytes)

        mock_ocr = MagicMock()
        mock_ocr.extract_text_from_document.return_value = [
            (1, "Scanned headline: Major news report extracted via OCR."),
            (2, "Page 2 continued story with full article text."),
        ]
        service = PDFService(ocr_service=mock_ocr)
        result = service.extract_pages(str(pdf_path))

        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert "headline" in result.pages[0].text
        mock_ocr.extract_text_from_document.assert_called_once()

    def test_empty_scanned_pdf_rejected(self, tmp_path, empty_pdf_bytes):
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(empty_pdf_bytes)

        mock_ocr = MagicMock()
        mock_ocr.extract_text_from_document.return_value = []
        service = PDFService(ocr_service=mock_ocr)

        with pytest.raises(ScannedPDFError) as excinfo:
            service.extract_pages(str(pdf_path))
        assert "blank" in excinfo.value.message.lower() or "scanned" in excinfo.value.message.lower()

    def test_invalid_pdf_rejected(self, tmp_path, not_a_pdf_bytes):
        pdf_path = tmp_path / "invalid.pdf"
        pdf_path.write_bytes(not_a_pdf_bytes)

        with pytest.raises(InvalidPDFError):
            PDFService().extract_pages(str(pdf_path))


class TestTextCleaning:
    def setup_method(self):
        self.service = TextCleaningService()

    def test_collapses_excessive_whitespace(self):
        assert self.service.clean("too    many     spaces") == "too many spaces"

    def test_collapses_repeated_blank_lines(self):
        cleaned = self.service.clean("para one\n\n\n\n\npara two")
        assert cleaned == "para one\n\npara two"

    def test_preserves_paragraph_boundaries(self):
        cleaned = self.service.clean("first paragraph\n\nsecond paragraph")
        assert "\n\n" in cleaned

    def test_repairs_hyphenated_line_breaks(self):
        assert self.service.clean("devel-\nopment") == "development"

    def test_strips_control_characters(self):
        assert self.service.clean("hello\x00\x08world") == "helloworld"

    def test_empty_input(self):
        assert self.service.clean("") == ""
        assert self.service.clean("   \n\n  ") == ""
