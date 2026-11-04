"""Vector search / global retrieval tests.

Verifies the repository query is global: joined on documents, filtered only
by COMPLETED status, ordered by cosine distance — with NO document_id filter.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.document import DocumentStatus
from app.repositories.chunk_repository import ChunkRepository, RetrievedChunk
from app.services.rag_service import RAGService


def make_row(**overrides) -> MagicMock:
    row = MagicMock()
    row.id = overrides.get("id", uuid.uuid4())
    row.document_id = overrides.get("document_id", uuid.uuid4())
    row.filename = overrides.get("filename", "newspaper_a.pdf")
    row.page_number = overrides.get("page_number", 3)
    row.chunk_index = overrides.get("chunk_index", 0)
    row.chunk_text = overrides.get("chunk_text", "AI developments reported today.")
    row.distance = overrides.get("distance", 0.21)
    return row


async def run_search(rows, **kwargs):
    db = AsyncMock()
    results = MagicMock()
    results.all.return_value = rows
    db.execute = AsyncMock(return_value=results)
    repository = ChunkRepository(db)
    chunks = await repository.similarity_search_global([0.1] * 1536, **kwargs)
    return chunks, db


class TestGlobalVectorSearch:
    async def test_returns_retrieved_chunks_with_metadata(self):
        rows = [make_row(distance=0.1), make_row(distance=0.3, filename="newspaper_b.pdf")]
        chunks, _ = await run_search(rows)

        assert len(chunks) == 2
        assert isinstance(chunks[0], RetrievedChunk)
        assert chunks[0].filename == "newspaper_a.pdf"
        assert chunks[1].filename == "newspaper_b.pdf"
        assert chunks[0].page_number == 3

    async def test_query_has_no_document_id_filter(self):
        """The global search must not filter by any individual document."""
        _, db = await run_search([make_row()])

        select_stmt = None
        for call in db.execute.call_args_list:
            stmt = call.args[0]
            compiled = str(stmt) if not hasattr(stmt, "compile") else str(stmt.compile())
            if "document_chunks" in compiled:
                select_stmt = compiled
                break

        assert select_stmt is not None
        # Filters only on document status, never on a specific document_id.
        assert "documents.status" in select_stmt
        where_clause = select_stmt.split("WHERE", 1)[1]
        assert "document_id =" not in where_clause
        assert "documents.id =" not in where_clause

    async def test_query_orders_by_distance_and_limits(self):
        _, db = await run_search([make_row()], top_k=7)

        compiled = None
        for call in db.execute.call_args_list:
            stmt = call.args[0]
            if hasattr(stmt, "compile") and "document_chunks" in str(stmt):
                compiled = str(stmt.compile())
        assert compiled is not None
        assert "ORDER BY" in compiled
        assert "<=>" in compiled  # cosine distance operator
        assert "LIMIT" in compiled

    async def test_distance_threshold_filters_results(self):
        rows = [make_row(distance=0.1), make_row(distance=0.9)]
        chunks, _ = await run_search(rows, max_distance=0.5)
        assert len(chunks) == 1
        assert chunks[0].distance == 0.1

    async def test_only_completed_status_in_query(self):
        _, db = await run_search([make_row()])
        compiled = ""
        for call in db.execute.call_args_list:
            stmt = call.args[0]
            if hasattr(stmt, "compile") and "document_chunks" in str(stmt):
                compiled = str(
                    stmt.compile(compile_kwargs={"literal_binds": False})
                )
        assert "status" in compiled


class TestCitationMetadata:
    def test_build_sources_traceable_to_chunks(self):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            document_id=uuid.uuid4(),
            filename="times.pdf",
            page_number=5,
            chunk_index=2,
            chunk_text="X" * 500,
            distance=0.2134,
        )
        sources = RAGService.build_sources([chunk])

        assert len(sources) == 1
        source = sources[0]
        assert source["document_id"] == str(chunk.document_id)
        assert source["chunk_id"] == str(chunk.chunk_id)
        assert source["filename"] == "times.pdf"
        assert source["page_number"] == 5
        assert source["relevance_distance"] == 0.2134
        assert len(source["excerpt"]) <= 300
