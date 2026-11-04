"""Shared test fixtures.

Unit tests never touch PostgreSQL, Redis, or OpenAI:
- OpenAI calls are mocked.
- Database-backed services are replaced through FastAPI dependency overrides.
"""

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fitz
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import get_db
from app.main import app
from app.models.document import Document, DocumentStatus


# --- PDF fixtures ------------------------------------------------------------


@pytest.fixture
def digital_pdf_bytes() -> bytes:
    """A small digital/searchable PDF with real extractable text."""
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page()
        text = (
            f"Page {page_number + 1}. Artificial intelligence continues to "
            "advance rapidly. Researchers announced new language models this "
            "week, and policymakers debated regulation of AI systems across "
            "several countries. " * 5
        )
        page.insert_text((72, 72), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def empty_pdf_bytes() -> bytes:
    """A PDF with pages but no extractable text (simulates a scanned PDF)."""
    document = fitz.open()
    document.new_page()
    document.new_page()
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def not_a_pdf_bytes() -> bytes:
    return b"This is definitely not a PDF file."


@pytest.fixture
def hindi_pdf_bytes() -> bytes:
    """A digital PDF with authentic Unicode Devanagari Hindi text."""
    import os

    document = fitz.open()
    font_candidates = [
        "C:/Windows/Fonts/Nirmala.ttc",
        "C:/Windows/Fonts/mangal.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((f for f in font_candidates if os.path.exists(f)), None)

    for page_number in range(2):
        page = document.new_page()
        text = (
            f"पृष्ठ {page_number + 1}. भारत की अर्थव्यवस्था तेज़ी से बढ़ रही है। "
            "भारतीय रिज़र्व बैंक ने रेपो रेट को 6.5 प्रतिशत पर बरकरार रखा है। "
            "उद्योग और कृषि क्षेत्र में निरंतर सुधार देखा जा रहा है। "
            "विशेषज्ञों का अनुमान है कि चालू वित्त वर्ष में जीडीपी वृद्धि दर 7 प्रतिशत रहेगी। "
        ) * 5
        if font_path:
            page.insert_font(fontname="hindi_font", fontfile=font_path)
            page.insert_textbox(
                fitz.Rect(50, 50, 550, 750),
                text,
                fontname="hindi_font",
                fontsize=11,
            )
        else:
            # Fallback if no local font is installed: insert text normally
            page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def legacy_font_pdf_bytes() -> bytes:
    """A legacy-encoded Hindi PDF that maps ASCII characters to Devanagari via Kruti Dev font."""
    document = fitz.open()
    for page_number in range(2):
        page = document.new_page()
        text = (
            f"Page {page_number + 1}. dksbZ Hkh O;fDr iz/kkuea=h th us dgk fd ns'k esa "
            "fodkl gks jgk gSA Hkkjrh; turk ikVhZ ds usrkvksa us cSBd dhA "
            "jk\"Vªifr Hkou esa fo'ks\"k dk;ZØe vk;ksftr fd;k x;kA "
        ) * 6
        page.insert_textbox(
            fitz.Rect(50, 50, 550, 750),
            text,
            fontsize=11,
        )
    data = document.tobytes()
    document.close()
    return data


# --- Model factories ---------------------------------------------------------


def make_document(**overrides) -> Document:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=uuid.uuid4(),
        filename="newspaper.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        file_path="/app/storage/newspapers/test.pdf",
        file_hash="a" * 64,
        file_size=1024,
        total_pages=4,
        total_chunks=12,
        status=DocumentStatus.COMPLETED,
        language="en",
        error_message=None,
        created_at=now,
        updated_at=now,
        processed_at=now,
    )
    defaults.update(overrides)
    document = Document()
    for key, value in defaults.items():
        setattr(document, key, value)
    return document


# --- FastAPI client ----------------------------------------------------------


@pytest.fixture
def mock_db():
    """An AsyncMock standing in for the AsyncSession dependency."""
    db = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.fixture
async def client(mock_db):
    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


def pdf_upload(data: bytes, filename: str = "newspaper.pdf"):
    return {"file": (filename, io.BytesIO(data), "application/pdf")}
