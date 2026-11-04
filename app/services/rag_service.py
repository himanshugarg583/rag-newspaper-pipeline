"""RAG orchestration: contextualize -> embed -> global retrieve -> answer."""

import asyncio
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.repositories.chunk_repository import RetrievedChunk
from app.services.language_service import LanguageService
from app.services.llm_service import LLMService
from app.services.vector_search_service import VectorSearchService

logger = get_logger(__name__)

EXCERPT_MAX_CHARS = 300

NO_CONTEXT_ANSWER_EN = (
    "I could not find relevant information in the uploaded newspapers to "
    "answer this question. Please make sure newspapers covering this topic "
    "have been uploaded and processed successfully."
)

NO_CONTEXT_ANSWER_HI = (
    "अपलोड किए गए समाचार पत्रों में इस प्रश्न का उत्तर देने के लिए प्रासंगिक जानकारी नहीं मिली। "
    "कृपया सुनिश्चित करें कि इस विषय से संबंधित समाचार पत्र अपलोड और प्रोसेस हो चुके हैं।"
)

NO_CONTEXT_ANSWER_HINGLISH = (
    "Upload kiye gaye newspapers me is question ka answer dene ke liye relevant jankari nahi mili. "
    "Kripya ensure karein ki is topic se related newspapers upload aur process ho chuke hain."
)
NO_CONTEXT_ANSWER = NO_CONTEXT_ANSWER_EN


@dataclass(frozen=True)
class RAGResult:
    answer: str
    chunks: list[RetrievedChunk]
    sources: list[dict]


class RAGService:
    def __init__(
        self,
        db: AsyncSession,
        vector_search_service: VectorSearchService | None = None,
        llm_service: LLMService | None = None,
    ):
        self.vector_search_service = vector_search_service or VectorSearchService(db)
        self.llm_service = llm_service or LLMService()

    @staticmethod
    def build_sources(chunks: list[RetrievedChunk]) -> list[dict]:
        """Build citation metadata traceable to actual retrieved chunks."""
        return [
            {
                "document_id": str(chunk.document_id),
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_id": str(chunk.chunk_id),
                "relevance_distance": round(chunk.distance, 4),
                "excerpt": chunk.chunk_text[:EXCERPT_MAX_CHARS],
            }
            for chunk in chunks
        ]

    async def answer_question(
        self,
        question: str,
        history: list[dict[str, str]],
    ) -> RAGResult:
        """Run the full RAG pipeline for a single user question.

        A fresh global vector search is performed for every question; the
        conversation history is only used to contextualize the query and to
        help the LLM understand follow-ups, never as a retrieval substitute.
        """
        search_query = await asyncio.to_thread(
            self.llm_service.contextualize_question, question, history
        )
        if search_query != question:
            logger.info("Contextualized follow-up question for retrieval")

        chunks = await self.vector_search_service.search(search_query)

        if not chunks:
            try:
                target_lang = self.llm_service.determine_target_language(question)
                if not isinstance(target_lang, str):
                    target_lang = LLMService.determine_target_language(question)
            except Exception:
                target_lang = LLMService.determine_target_language(question)

            if target_lang == "Hindi":
                fallback = NO_CONTEXT_ANSWER_HI
            elif target_lang == "Hinglish":
                fallback = NO_CONTEXT_ANSWER_HINGLISH
            else:
                fallback = NO_CONTEXT_ANSWER_EN
            return RAGResult(answer=fallback, chunks=[], sources=[])

        answer = await asyncio.to_thread(
            self.llm_service.generate_answer, question, chunks, history
        )
        return RAGResult(
            answer=answer,
            chunks=chunks,
            sources=self.build_sources(chunks),
        )
