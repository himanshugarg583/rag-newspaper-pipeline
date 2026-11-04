"""Document API tests: upload validation, details, deletion, reprocess."""

import io
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.controllers.document_controller import get_document_service
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    FileSizeExceededError,
)
from app.main import app
from app.models.document import DocumentStatus
from app.services.document_service import DocumentService
from tests.conftest import make_document


@pytest.fixture
def mock_service():
    service = MagicMock(spec=DocumentService)
    app.dependency_overrides[get_document_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_document_service, None)


def upload_files(data: bytes, filename="newspaper.pdf", content_type="application/pdf"):
    return {"file": (filename, io.BytesIO(data), content_type)}


class TestUploadValidation:
    async def test_upload_accepts_valid_pdf(self, client, mock_service, digital_pdf_bytes):
        document = make_document(status=DocumentStatus.PROCESSING)
        mock_service.upload = AsyncMock(return_value=document)

        response = await client.post(
            "/api/v1/documents/upload", files=upload_files(digital_pdf_bytes)
        )

        assert response.status_code == 202
        body = response.json()
        assert body["document_id"] == str(document.id)
        assert body["status"] == "PROCESSING"

    async def test_upload_rejects_non_pdf_extension(self, client, mock_db):
        # Use the real service to exercise validation logic.
        service = DocumentService(mock_db)
        app.dependency_overrides[get_document_service] = lambda: service
        try:
            response = await client.post(
                "/api/v1/documents/upload",
                files=upload_files(b"hello", filename="notes.txt", content_type="text/plain"),
            )
        finally:
            app.dependency_overrides.pop(get_document_service, None)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_FILE_TYPE"

    async def test_upload_rejects_fake_pdf_content(self, client, mock_db, not_a_pdf_bytes):
        service = DocumentService(mock_db)
        app.dependency_overrides[get_document_service] = lambda: service
        try:
            response = await client.post(
                "/api/v1/documents/upload", files=upload_files(not_a_pdf_bytes)
            )
        finally:
            app.dependency_overrides.pop(get_document_service, None)

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "INVALID_FILE_TYPE"

    async def test_upload_rejects_oversized_file(self, client, mock_service):
        mock_service.upload = AsyncMock(
            side_effect=FileSizeExceededError("File exceeds the maximum upload size.")
        )
        response = await client.post(
            "/api/v1/documents/upload", files=upload_files(b"%PDF-1.7 big")
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "FILE_SIZE_EXCEEDED"

    async def test_upload_rejects_duplicate(self, client, mock_service):
        mock_service.upload = AsyncMock(
            side_effect=DuplicateDocumentError("This PDF has already been uploaded.")
        )
        response = await client.post(
            "/api/v1/documents/upload", files=upload_files(b"%PDF-1.7 dup")
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "DUPLICATE_DOCUMENT"


class TestDocumentEndpoints:
    async def test_list_documents(self, client, mock_service):
        documents = [make_document(), make_document(status=DocumentStatus.PROCESSING)]
        mock_service.list_documents = AsyncMock(return_value=(documents, 2))

        response = await client.get("/api/v1/documents?page=1&page_size=10")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2
        assert body["total_pages"] == 1

    async def test_get_document_details(self, client, mock_service):
        document = make_document(
            status=DocumentStatus.FAILED, error_message="Scanned PDF rejected."
        )
        mock_service.get_document = AsyncMock(return_value=document)

        response = await client.get(f"/api/v1/documents/{document.id}")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "FAILED"
        assert body["error_message"] == "Scanned PDF rejected."
        assert body["total_pages"] == 4
        assert body["total_chunks"] == 12

    async def test_get_document_not_found(self, client, mock_service):
        mock_service.get_document = AsyncMock(
            side_effect=DocumentNotFoundError("Document not found.")
        )
        response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    async def test_delete_document(self, client, mock_service):
        mock_service.delete_document = AsyncMock()
        response = await client.delete(f"/api/v1/documents/{uuid.uuid4()}")
        assert response.status_code == 200
        assert response.json()["success"] is True
        mock_service.delete_document.assert_awaited_once()

    async def test_reprocess_document(self, client, mock_service):
        document = make_document(status=DocumentStatus.PROCESSING)
        mock_service.reprocess_document = AsyncMock(return_value=document)

        response = await client.post(f"/api/v1/documents/{document.id}/reprocess")

        assert response.status_code == 200
        assert response.json()["status"] == "PROCESSING"


class TestDeleteServiceLogic:
    async def test_delete_removes_file_after_commit(self, mock_db, tmp_path):
        pdf_file = tmp_path / "stored.pdf"
        pdf_file.write_bytes(b"%PDF-1.7")
        document = make_document(file_path=str(pdf_file))

        service = DocumentService(mock_db)
        service.document_repository.get_by_id = AsyncMock(return_value=document)
        service.document_repository.delete = AsyncMock()

        await service.delete_document(document.id)

        service.document_repository.delete.assert_awaited_once_with(document)
        mock_db.commit.assert_awaited_once()
        assert not pdf_file.exists()

    async def test_delete_handles_missing_file_gracefully(self, mock_db):
        document = make_document(file_path="/nonexistent/path/file.pdf")
        service = DocumentService(mock_db)
        service.document_repository.get_by_id = AsyncMock(return_value=document)
        service.document_repository.delete = AsyncMock()

        # Must not raise even though the file does not exist.
        await service.delete_document(document.id)
        mock_db.commit.assert_awaited_once()
