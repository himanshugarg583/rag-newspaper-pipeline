"""Document API endpoints (thin controllers delegating to services)."""

import uuid

from fastapi import APIRouter, Depends, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.common import MessageResponse
from app.schemas.document import (
    DocumentListResponse,
    DocumentReprocessResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService
from app.utils.response_utils import total_pages

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(db: AsyncSession = Depends(get_db)) -> DocumentService:
    return DocumentService(db)


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile,
    service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    document = await service.upload(file)
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        message="Upload accepted. The newspaper is being processed in the background.",
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    documents, total = await service.list_documents(page, page_size)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in documents],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages(total, page_size),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    document = await service.get_document(document_id)
    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", response_model=MessageResponse)
async def delete_document(
    document_id: uuid.UUID,
    service: DocumentService = Depends(get_document_service),
) -> MessageResponse:
    await service.delete_document(document_id)
    return MessageResponse(message="Document and associated data deleted.")


@router.post("/{document_id}/reprocess", response_model=DocumentReprocessResponse)
async def reprocess_document(
    document_id: uuid.UUID,
    service: DocumentService = Depends(get_document_service),
) -> DocumentReprocessResponse:
    document = await service.reprocess_document(document_id)
    return DocumentReprocessResponse(
        document_id=document.id,
        status=document.status,
        message="Document reprocessing has been queued.",
    )
