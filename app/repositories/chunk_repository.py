"""Chunk repository, including global pgvector similarity search.

The global search intentionally has NO document_id filter: every chat question
searches across all COMPLETED newspapers together. The ORDER BY on the raw
`embedding <=> :query_embedding` expression keeps the query HNSW-compatible.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.repositories.base_repository import BaseRepository

COMMON_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "tell", "news", "what", "when", "where", "which",
    "who", "whom", "this", "that", "these", "those", "kya", "hai",
    "hain", "tha", "the", "thi", "ka", "ke", "ki", "ko", "se", "me",
    "mein", "par", "pe", "aur", "ya", "batao", "samachar", "khabar",
    "latest", "today", "yesterday", "article", "report", "reports"
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_index: int
    chunk_text: str
    distance: float


class ChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    def __init__(self, db: AsyncSession):
        super().__init__(db)

    async def count_for_document(self, document_id: uuid.UUID) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        return int(result.scalar_one())

    async def similarity_search_global(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        max_distance: float | None = None,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """Hybrid Cosine-distance + Keyword search across ALL completed documents (no doc filter)."""
        k = top_k or settings.retrieval_top_k

        # Transaction-local HNSW recall tuning; resets at transaction end.
        if settings.hnsw_ef_search:
            await self.db.execute(
                text("SELECT set_config('hnsw.ef_search', :ef, true)").bindparams(
                    ef=str(int(settings.hnsw_ef_search))
                )
            )

        distance = DocumentChunk.embedding.cosine_distance(query_embedding).label(
            "distance"
        )
        vec_stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                Document.filename,
                DocumentChunk.page_number,
                DocumentChunk.chunk_index,
                DocumentChunk.chunk_text,
                distance,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.status == DocumentStatus.COMPLETED)
            .order_by(distance)
            .limit(k * 2)
        )

        vec_result = await self.db.execute(vec_stmt)
        vec_rows = vec_result.all()

        # Keyword matching for named entities / specific search terms
        kw_rows = []
        if query_text:
            raw_tokens = re.findall(r"[\u0900-\u097F]+|[a-zA-Z0-9]+", query_text.lower())
            keywords = [w for w in raw_tokens if len(w) >= 3 and w not in COMMON_STOPWORDS]
            if keywords:
                # Up to 8 most specific terms
                conds = [DocumentChunk.chunk_text.ilike(f"%{kw}%") for kw in keywords[:8]]
                kw_stmt = (
                    select(
                        DocumentChunk.id,
                        DocumentChunk.document_id,
                        Document.filename,
                        DocumentChunk.page_number,
                        DocumentChunk.chunk_index,
                        DocumentChunk.chunk_text,
                        distance,
                    )
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(Document.status == DocumentStatus.COMPLETED)
                    .where(or_(*conds))
                    .limit(k * 2)
                )
                kw_res = await self.db.execute(kw_stmt)
                for r in kw_res.all():
                    match_count = sum(1 for kw in keywords if kw in r.chunk_text.lower())
                    kw_rows.append((r, match_count))

        # Reciprocal Rank Fusion (RRF) to combine dense vector and keyword matches
        scores: dict[uuid.UUID, float] = {}
        chunks_map: dict[uuid.UUID, tuple] = {}

        for rank, row in enumerate(vec_rows):
            cid = row.id
            chunks_map[cid] = row
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (60 + rank + 1))

        for rank, (row, match_count) in enumerate(kw_rows):
            cid = row.id
            if cid not in chunks_map:
                chunks_map[cid] = row
            # Boost exact keyword matches
            scores[cid] = scores.get(cid, 0.0) + (2.0 / (60 + rank + 1)) * match_count

        sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:k]

        chunks = [
            RetrievedChunk(
                chunk_id=chunks_map[cid].id,
                document_id=chunks_map[cid].document_id,
                filename=chunks_map[cid].filename,
                page_number=chunks_map[cid].page_number,
                chunk_index=chunks_map[cid].chunk_index,
                chunk_text=chunks_map[cid].chunk_text,
                distance=float(getattr(chunks_map[cid], "distance", 0.0)),
            )
            for cid in sorted_ids
        ]

        threshold = (
            max_distance
            if max_distance is not None
            else (settings.retrieval_max_distance or None)
        )
        if threshold:
            chunks = [c for c in chunks if c.distance <= threshold]
        return chunks
