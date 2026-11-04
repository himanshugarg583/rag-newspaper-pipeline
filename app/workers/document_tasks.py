"""Celery tasks for asynchronous newspaper processing.

Workers use the synchronous SQLAlchemy engine. The pipeline is idempotent:
existing chunks for a document are deleted before new chunks are inserted, so
retries and reprocessing never produce duplicates, and failures never leave a
document marked COMPLETED with partial data.
"""

import uuid
from datetime import datetime, timezone

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded, TimeLimitExceeded
from sqlalchemy import delete, select

from app.core.celery_app import celery_app
from app.core.config import settings
from app.core.database import SyncSessionLocal
from app.core.exceptions import (
    AppException,
    CorruptedPDFTextError,
    InvalidPDFError,
    OpenAIServiceError,
    ScannedPDFError,
)
from app.core.logging import get_logger, setup_logging
from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.language_service import LanguageService
from app.services.pdf_service import PDFService
from app.services.text_cleaning_service import TextCleaningService

setup_logging()
logger = get_logger(__name__)

# Errors worth retrying (infrastructure / rate limiting), not content errors.
RETRYABLE_ERRORS = (OpenAIServiceError, ConnectionError, TimeoutError)


def _mark_failed(document_id: uuid.UUID, error_message: str) -> None:
    """Mark a document FAILED and remove any partially inserted chunks."""
    with SyncSessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            return
        db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        document.status = DocumentStatus.FAILED
        document.error_message = error_message[:2000]
        document.total_chunks = 0
        db.commit()


@celery_app.task(
    bind=True,
    name="documents.process",
    max_retries=3,
    default_retry_delay=30,
    retry_backoff=True,
    retry_backoff_max=300,
)
def process_document(self: Task, document_id_str: str) -> dict:
    """Full ingestion pipeline: extract -> clean -> chunk -> embed -> persist."""
    document_id = uuid.UUID(document_id_str)
    logger.info("Processing document %s", document_id)

    with SyncSessionLocal() as db:
        document = db.get(Document, document_id)
        if document is None:
            logger.warning("Document %s no longer exists; skipping", document_id)
            return {"document_id": document_id_str, "status": "SKIPPED"}
        file_path = document.file_path

    try:
        # 1. Extract text page by page (digital PDFs only; no OCR).
        pdf_service = PDFService()
        extraction = pdf_service.extract_pages(file_path)

        # 2. Clean each page's text.
        cleaning_service = TextCleaningService()
        cleaned_pages = [
            (page.page_number, cleaning_service.clean(page.text))
            for page in extraction.pages
        ]
        cleaned_pages = [(num, txt) for num, txt in cleaned_pages if txt]
        if not cleaned_pages:
            raise ScannedPDFError(
                "No usable text remained after cleaning; the PDF may be "
                "scanned or image-only."
            )

        # 3. Page-wise chunking (page citations stay accurate).
        chunking_service = ChunkingService()
        chunks = chunking_service.chunk_pages(cleaned_pages)
        if not chunks:
            raise ScannedPDFError("The PDF produced no text chunks.")

        # 4. Detect document-level language
        combined_cleaned_text = "\n".join(txt for _, txt in cleaned_pages)
        doc_language = LanguageService.detect_language(combined_cleaned_text)

        # 5. Batched embedding generation with retry/backoff.
        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_texts([c.text for c in chunks])

        # 6. Persist chunks + vectors and mark COMPLETED in one transaction.
        with SyncSessionLocal() as db:
            document = db.get(Document, document_id)
            if document is None:
                return {"document_id": document_id_str, "status": "SKIPPED"}

            # Idempotency: remove any chunks from earlier attempts.
            db.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id
                )
            )
            for chunk, embedding in zip(chunks, embeddings):
                db.add(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_text=chunk.text,
                        embedding=embedding,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        language=chunk.language,
                    )
                )

            document.total_pages = extraction.total_pages
            document.total_chunks = len(chunks)
            document.status = DocumentStatus.COMPLETED
            document.language = doc_language
            document.error_message = None
            document.embedding_model = settings.openai_embedding_model
            document.processed_at = datetime.now(timezone.utc)
            db.commit()

        logger.info(
            "Document %s completed: %d pages, %d chunks, language=%s",
            document_id,
            extraction.total_pages,
            len(chunks),
            doc_language,
        )
        return {
            "document_id": document_id_str,
            "status": "COMPLETED",
            "total_chunks": len(chunks),
            "language": doc_language,
        }

    except (InvalidPDFError, ScannedPDFError, CorruptedPDFTextError) as exc:
        # Permanent content errors: fail immediately, no retry.
        logger.warning("Document %s rejected: %s", document_id, exc.message)
        _mark_failed(document_id, exc.message)
        return {"document_id": document_id_str, "status": "FAILED"}

    except RETRYABLE_ERRORS as exc:
        message = exc.message if isinstance(exc, AppException) else str(exc)
        if self.request.retries >= self.max_retries:
            logger.error(
                "Document %s failed after %d retries", document_id, self.request.retries
            )
            _mark_failed(document_id, f"Processing failed after retries: {message}")
            return {"document_id": document_id_str, "status": "FAILED"}
        logger.warning(
            "Transient error for document %s (retry %d): %s",
            document_id,
            self.request.retries + 1,
            message,
        )
        raise self.retry(exc=exc)

    except (SoftTimeLimitExceeded, TimeLimitExceeded) as exc:
        logger.error("Processing timed out for document %s: %s", document_id, exc)
        _mark_failed(
            document_id,
            "Processing timed out. The document contains too many complex scanned pages.",
        )
        return {"document_id": document_id_str, "status": "FAILED"}

    except Exception as exc:
        logger.exception("Unexpected error processing document %s", document_id)
        _mark_failed(document_id, f"Unexpected processing error: {exc}")
        return {"document_id": document_id_str, "status": "FAILED"}

    except BaseException as exc:
        logger.exception("Fatal worker interruption for document %s: %s", document_id, exc)
        _mark_failed(document_id, f"Processing interrupted: {exc}")
        raise
