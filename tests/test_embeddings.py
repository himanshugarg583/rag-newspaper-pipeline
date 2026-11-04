"""Embedding service tests with mocked OpenAI API."""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.core.exceptions import EmbeddingDimensionError
from app.services.embedding_service import EmbeddingService

DIMS = settings.openai_embedding_dimensions


def make_mock_client(dimensions: int = DIMS) -> MagicMock:
    client = MagicMock()

    def create(model, input, dimensions=dimensions, **kwargs):
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[0.1] * DIMS if dimensions == DIMS else [0.1] * dimensions)
            for _ in input
        ]
        return response

    client.embeddings.create.side_effect = create
    return client


class TestEmbeddingService:
    def test_embed_texts_with_mocked_openai(self):
        client = make_mock_client()
        service = EmbeddingService(client=client)

        embeddings = service.embed_texts(["chunk one", "chunk two", "chunk three"])

        assert len(embeddings) == 3
        assert all(len(e) == DIMS for e in embeddings)
        client.embeddings.create.assert_called()

    def test_embed_query_returns_single_vector(self):
        service = EmbeddingService(client=make_mock_client())
        embedding = service.embed_query("what happened today?")
        assert len(embedding) == DIMS

    def test_batching_reduces_api_calls(self):
        client = make_mock_client()
        service = EmbeddingService(client=client)
        service.batch_size = 10

        service.embed_texts([f"text {i}" for i in range(25)])

        assert client.embeddings.create.call_count == 3  # 10 + 10 + 5

    def test_dimension_mismatch_raises(self):
        client = MagicMock()
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1] * 768)]  # wrong dimensions
        client.embeddings.create.return_value = response
        service = EmbeddingService(client=client)

        with pytest.raises(EmbeddingDimensionError):
            service.embed_texts(["some text"])

    def test_empty_input_returns_empty(self):
        service = EmbeddingService(client=make_mock_client())
        assert service.embed_texts([]) == []
