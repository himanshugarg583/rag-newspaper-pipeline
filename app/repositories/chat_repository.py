"""Chat session and message repositories."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage, MessageRole
from app.models.chat_session import ChatSession
from app.repositories.base_repository import BaseRepository


class ChatRepository(BaseRepository[ChatSession]):
    model = ChatSession

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def create_session(self) -> ChatSession:
        session = ChatSession()
        return await self.add(session)

    async def get_messages(
        self, session_id: uuid.UUID, limit: int | None = None
    ) -> list[ChatMessage]:
        """Return messages in chronological order. When limit is set, return
        the most recent `limit` messages (still chronological)."""
        stmt = select(ChatMessage).where(ChatMessage.session_id == session_id)
        if limit:
            stmt = stmt.order_by(ChatMessage.created_at.desc()).limit(limit)
            result = await self.db.execute(stmt)
            return list(reversed(result.scalars().all()))
        stmt = stmt.order_by(ChatMessage.created_at.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def add_message(
        self,
        session_id: uuid.UUID,
        role: MessageRole,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
        )
        self.db.add(message)
        await self.db.flush()
        return message
