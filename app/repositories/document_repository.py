"""Document repository: async access for the API layer."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentStatus
from app.repositories.base_repository import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def get_by_hash(self, file_hash: str) -> Document | None:
        result = await self.db.execute(
            select(Document).where(Document.file_hash == file_hash)
        )
        return result.scalar_one_or_none()

    async def list_paginated(
        self, offset: int, limit: int
    ) -> tuple[list[Document], int]:
        total_result = await self.db.execute(
            select(func.count()).select_from(Document)
        )
        total = int(total_result.scalar_one())

        result = await self.db.execute(
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def create(
        self,
        *,
        filename: str,
        stored_filename: str,
        file_path: str,
        file_hash: str,
        file_size: int,
    ) -> Document:
        document = Document(
            filename=filename,
            stored_filename=stored_filename,
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            status=DocumentStatus.PROCESSING,
        )
        return await self.add(document)

    async def mark_processing(self, document: Document) -> Document:
        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        document.processed_at = None
        await self.db.flush()
        return document

    async def update_status(
        self,
        document: Document,
        status: DocumentStatus,
        error_message: str | None = None,
    ) -> Document:
        document.status = status
        document.error_message = error_message
        if status == DocumentStatus.COMPLETED:
            document.processed_at = datetime.now(timezone.utc)
        await self.db.flush()
        return document
