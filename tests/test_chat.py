"""Chat endpoint, history, and RAG pipeline tests (OpenAI fully mocked)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.controllers.chat_controller import get_chat_service
from app.core.exceptions import ChatSessionNotFoundError
from app.main import app
from app.models.chat_message import ChatMessage, MessageRole
from app.repositories.chunk_repository import RetrievedChunk
from app.services.chat_service import ChatService
from app.services.rag_service import NO_CONTEXT_ANSWER, RAGService


def make_chunk(**overrides) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=overrides.get("chunk_id", uuid.uuid4()),
        document_id=overrides.get("document_id", uuid.uuid4()),
        filename=overrides.get("filename", "times_of_india.pdf"),
        page_number=overrides.get("page_number", 5),
        chunk_index=overrides.get("chunk_index", 0),
        chunk_text=overrides.get("chunk_text", "AI adoption increased this year."),
        distance=overrides.get("distance", 0.21),
    )


def make_message(role: MessageRole, content: str, sources=None) -> ChatMessage:
    message = ChatMessage()
    message.id = uuid.uuid4()
    message.role = role
    message.content = content
    message.sources = sources
    message.created_at = datetime.now(timezone.utc)
    return message


@pytest.fixture
def mock_chat_service():
    service = MagicMock(spec=ChatService)
    app.dependency_overrides[get_chat_service] = lambda: service
    yield service
    app.dependency_overrides.pop(get_chat_service, None)


class TestChatEndpoint:
    async def test_ask_question_returns_answer_and_citations(
        self, client, mock_chat_service
    ):
        session_id = uuid.uuid4()
        chunk = make_chunk()
        sources = RAGService.build_sources([chunk])
        mock_chat_service.ask = AsyncMock(
            return_value=(session_id, "AI adoption increased this year.", sources)
        )

        response = await client.post(
            "/api/v1/chat",
            json={"session_id": None, "question": "What happened with AI?"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == str(session_id)
        assert body["question"] == "What happened with AI?"
        assert body["answer"] == "AI adoption increased this year."
        # Sources are stored internally but not returned in the chat response.
        assert "sources" not in body

    async def test_ask_question_validation(self, client, mock_chat_service):
        response = await client.post("/api/v1/chat", json={"question": ""})
        assert response.status_code == 422

    async def test_invalid_session_returns_404(self, client, mock_chat_service):
        mock_chat_service.ask = AsyncMock(
            side_effect=ChatSessionNotFoundError("Chat session not found.")
        )
        response = await client.post(
            "/api/v1/chat",
            json={"session_id": str(uuid.uuid4()), "question": "hello?"},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "CHAT_SESSION_NOT_FOUND"


class TestChatHistory:
    async def test_history_returned_chronologically(self, client, mock_chat_service):
        session_id = uuid.uuid4()
        messages = [
            make_message(MessageRole.USER, "What about the economy?"),
            make_message(MessageRole.ASSISTANT, "The economy grew.", sources=[]),
        ]
        mock_chat_service.get_history = AsyncMock(return_value=messages)

        response = await client.get(f"/api/v1/chat/history/{session_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == str(session_id)
        assert [m["role"] for m in body["messages"]] == ["USER", "ASSISTANT"]

    async def test_history_unknown_session_404(self, client, mock_chat_service):
        mock_chat_service.get_history = AsyncMock(
            side_effect=ChatSessionNotFoundError("Chat session not found.")
        )
        response = await client.get(f"/api/v1/chat/history/{uuid.uuid4()}")
        assert response.status_code == 404


class TestRAGPipeline:
    async def test_every_question_triggers_fresh_global_search(self, mock_db):
        """Even with history available, retrieval must run for each question."""
        vector_search = MagicMock()
        vector_search.search = AsyncMock(return_value=[make_chunk()])
        llm = MagicMock()
        llm.contextualize_question.return_value = "standalone question"
        llm.generate_answer.return_value = "An answer."

        rag = RAGService(mock_db, vector_search_service=vector_search, llm_service=llm)

        history = [
            {"role": "user", "content": "What happened in the Indian economy?"},
            {"role": "assistant", "content": "It grew by 7%."},
        ]
        result = await rag.answer_question("What are its effects on employment?", history)

        vector_search.search.assert_awaited_once_with("standalone question")
        llm.contextualize_question.assert_called_once()
        assert result.answer == "An answer."
        assert len(result.sources) == 1

    async def test_no_context_returns_honest_answer(self, mock_db):
        vector_search = MagicMock()
        vector_search.search = AsyncMock(return_value=[])
        llm = MagicMock()
        llm.contextualize_question.return_value = "unanswerable question"

        rag = RAGService(mock_db, vector_search_service=vector_search, llm_service=llm)
        result = await rag.answer_question("unanswerable question", [])

        assert result.answer == NO_CONTEXT_ANSWER
        assert result.sources == []
        llm.generate_answer.assert_not_called()

    async def test_standalone_question_skips_contextualization_history(self, mock_db):
        vector_search = MagicMock()
        vector_search.search = AsyncMock(return_value=[make_chunk()])
        llm = MagicMock()
        llm.contextualize_question.return_value = "What is new in AI?"
        llm.generate_answer.return_value = "Answer."

        rag = RAGService(mock_db, vector_search_service=vector_search, llm_service=llm)
        await rag.answer_question("What is new in AI?", [])

        vector_search.search.assert_awaited_once_with("What is new in AI?")


class TestChatServiceSessions:
    async def test_new_session_created_when_session_id_null(self, mock_db):
        service = ChatService(mock_db, rag_service=MagicMock())
        new_session = MagicMock()
        new_session.id = uuid.uuid4()
        service.chat_repository.create_session = AsyncMock(return_value=new_session)
        service.chat_repository.get_messages = AsyncMock(return_value=[])
        service.chat_repository.add_message = AsyncMock()
        service.rag_service.answer_question = AsyncMock(
            return_value=MagicMock(answer="hi", sources=[])
        )

        session_id, answer, sources = await service.ask("What's new?", None)

        assert session_id == new_session.id
        service.chat_repository.create_session.assert_awaited_once()
        assert service.chat_repository.add_message.await_count == 2
        mock_db.commit.assert_awaited_once()

    async def test_unknown_session_raises(self, mock_db):
        service = ChatService(mock_db, rag_service=MagicMock())
        service.chat_repository.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(ChatSessionNotFoundError):
            await service.ask("question", uuid.uuid4())


class TestLLMService:
    def test_generate_answer_strips_source_markers(self):
        from app.services.llm_service import LLMService

        mock_openai = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "The library uses AI for search [Source 1] and classification [Source 2, Source 3]."
        )
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        service = LLMService(client=mock_openai)
        chunk = make_chunk(chunk_text="Some text")
        answer = service.generate_answer("What does library use?", [chunk], [])

        assert "[Source" not in answer
        assert answer == "The library uses AI for search and classification."

