"""Document service: upload workflow, listing, deletion, reprocessing."""

import uuid

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DocumentNotReprocessableError,
    DuplicateDocumentError,
    FileSizeExceededError,
    InvalidFileTypeError,
)
from app.core.logging import get_logger
from app.core.security import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    generate_stored_filename,
    resolve_upload_path,
    sanitize_filename,
)
from app.models.document import Document, DocumentStatus
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.utils.file_utils import delete_file_safely, save_bytes
from app.utils.hash_utils import sha256_bytes

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.document_repository = DocumentRepository(db)
        self.chunk_repository = ChunkRepository(db)

    async def upload(self, file: UploadFile) -> Document:
        """Validate, store, and register an uploaded PDF, then enqueue processing."""
        original_filename = sanitize_filename(file.filename or "upload.pdf")

        # 1. Extension and MIME type validation.
        extension = (
            "." + original_filename.rsplit(".", 1)[-1].lower()
            if "." in original_filename
            else ""
        )
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidFileTypeError("Only PDF files are accepted.")
        if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
            raise InvalidFileTypeError(
                f"Invalid content type {file.content_type!r}; expected application/pdf."
            )

        # 2. Size validation (read once into memory, bounded by max size).
        data = await file.read()
        if len(data) > settings.max_upload_size_bytes:
            raise FileSizeExceededError(
                f"File exceeds the maximum upload size of "
                f"{settings.max_upload_size_mb} MB."
            )
        if not data.startswith(PDF_MAGIC):
            raise InvalidFileTypeError("The uploaded file is not a valid PDF.")

        # 3. Duplicate detection via SHA-256.
        file_hash = sha256_bytes(data)
        existing = await self.document_repository.get_by_hash(file_hash)
        if existing is not None:
            raise DuplicateDocumentError(
                f"This PDF has already been uploaded (document {existing.id})."
            )

        # 4. Store with a unique safe filename.
        stored_filename = generate_stored_filename(original_filename)
        file_path = resolve_upload_path(settings.upload_dir, stored_filename)
        save_bytes(file_path, data)

        # 5. Create DB record and enqueue background processing.
        try:
            document = await self.document_repository.create(
                filename=original_filename,
                stored_filename=stored_filename,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size=len(data),
            )
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            delete_file_safely(str(file_path))
            raise

        self._enqueue_processing(document.id)
        logger.info("Uploaded document %s (%s)", document.id, original_filename)
        return document

    @staticmethod
    def _enqueue_processing(document_id: uuid.UUID) -> None:
        # Imported lazily to avoid a circular import with the worker module.
        from app.workers.document_tasks import process_document

        process_document.delay(str(document_id))

    async def list_documents(
        self, page: int, page_size: int
    ) -> tuple[list[Document], int]:
        offset = (page - 1) * page_size
        return await self.document_repository.list_paginated(offset, page_size)

    async def get_document(self, document_id: uuid.UUID) -> Document:
        document = await self.document_repository.get_by_id(document_id)
        if document is None:
            raise DocumentNotFoundError(f"Document {document_id} not found.")
        return document

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Delete the document record, its chunks (cascade), and the PDF file."""
        document = await self.get_document(document_id)
        file_path = document.file_path
        try:
            await self.document_repository.delete(document)
            await self.db.commit()
        except Exception:
            await self.db.rollback()
            raise
        # Remove the file only after the DB transaction succeeded.
        delete_file_safely(file_path)
        logger.info("Deleted document %s", document_id)

    async def reprocess_document(self, document_id: uuid.UUID) -> Document:
        """Re-run processing for a failed or completed document (idempotent)."""
        document = await self.get_document(document_id)
        if document.status == DocumentStatus.PROCESSING:
            raise DocumentNotReprocessableError(
                "Document is already being processed."
            )
        await self.document_repository.mark_processing(document)
        await self.db.commit()
        self._enqueue_processing(document.id)
        logger.info("Reprocessing document %s", document_id)
        return document
