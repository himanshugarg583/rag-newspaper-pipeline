"""Vector search service: embeds the query and runs the global search."""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.repositories.chunk_repository import ChunkRepository, RetrievedChunk
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class VectorSearchService:
    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService | None = None,
    ):
        self.chunk_repository = ChunkRepository(db)
        self.embedding_service = embedding_service or EmbeddingService()

    async def search(
        self, query: str, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Embed the query and search globally across all completed newspapers."""
        # The OpenAI client is synchronous; run it off the event loop.
        query_embedding = await asyncio.to_thread(
            self.embedding_service.embed_query, query
        )
        chunks = await self.chunk_repository.similarity_search_global(
            query_embedding=query_embedding,
            top_k=top_k or settings.retrieval_top_k,
            query_text=query,
        )
        logger.info("Global vector search returned %d chunks", len(chunks))
        return chunks
