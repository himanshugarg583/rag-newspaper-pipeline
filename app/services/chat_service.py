"""Chat service: session management, history, and RAG invocation."""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ChatSessionNotFoundError
from app.core.logging import get_logger
from app.models.chat_message import ChatMessage, MessageRole
from app.models.chat_session import ChatSession
from app.repositories.chat_repository import ChatRepository
from app.services.rag_service import RAGService

logger = get_logger(__name__)


class ChatService:
    def __init__(self, db: AsyncSession, rag_service: RAGService | None = None):
        self.db = db
        self.chat_repository = ChatRepository(db)
        self.rag_service = rag_service or RAGService(db)

    async def _get_or_create_session(
        self, session_id: uuid.UUID | None
    ) -> ChatSession:
        if session_id is None:
            session = await self.chat_repository.create_session()
            logger.info("Created chat session %s", session.id)
            return session
        session = await self.chat_repository.get_by_id(session_id)
        if session is None:
            raise ChatSessionNotFoundError(f"Chat session {session_id} not found.")
        return session

    @staticmethod
    def _to_llm_history(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [
            {
                "role": "user" if m.role == MessageRole.USER else "assistant",
                "content": m.content,
            }
            for m in messages
        ]

    async def ask(
        self, question: str, session_id: uuid.UUID | None
    ) -> tuple[uuid.UUID, str, list[dict]]:
        """Handle one chat turn; returns (session_id, answer, sources)."""
        session = await self._get_or_create_session(session_id)

        recent_messages = await self.chat_repository.get_messages(
            session.id, limit=settings.chat_history_max_messages
        )
        history = self._to_llm_history(recent_messages)

        result = await self.rag_service.answer_question(question, history)

        await self.chat_repository.add_message(
            session.id, MessageRole.USER, question
        )
        await self.chat_repository.add_message(
            session.id, MessageRole.ASSISTANT, result.answer, sources=result.sources
        )
        await self.db.commit()

        return session.id, result.answer, result.sources

    async def get_history(self, session_id: uuid.UUID) -> list[ChatMessage]:
        session = await self.chat_repository.get_by_id(session_id)
        if session is None:
            raise ChatSessionNotFoundError(f"Chat session {session_id} not found.")
        return await self.chat_repository.get_messages(session_id)
