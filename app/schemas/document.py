"""Document API schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus


class DocumentUploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    message: str


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_size: int
    total_pages: int | None
    total_chunks: int | None
    status: DocumentStatus
    error_message: str | None
    language: str | None = None
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentReprocessResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    message: str
