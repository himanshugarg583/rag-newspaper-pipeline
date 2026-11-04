"""OpenAI embedding generation with batching, retries, and validation.

Uses the synchronous OpenAI client so it is usable from both Celery workers
and (via thread executor) async request handlers. The chat flow embeds a
single query, which is fast enough to run in a worker thread.
"""

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import EmbeddingDimensionError, OpenAIServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


class EmbeddingService:
    def __init__(self, client: OpenAI | None = None):
        self._client = client
        self.model = settings.openai_embedding_model
        self.dimensions = settings.openai_embedding_dimensions
        self.batch_size = settings.embedding_batch_size

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.openai_request_timeout,
                max_retries=0,  # tenacity handles retries with backoff
            )
        return self._client

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        # OpenAI returns embeddings in input order.
        return [item.embedding for item in response.data]

    def _validate(self, embeddings: list[list[float]]) -> None:
        for embedding in embeddings:
            if len(embedding) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"Expected embedding dimension {self.dimensions}, "
                    f"got {len(embedding)}."
                )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed many texts in batches; validates dimensions before returning."""
        if not texts:
            return []
        embeddings: list[list[float]] = []
        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                embeddings.extend(self._embed_batch(batch))
        except _RETRYABLE as exc:
            raise OpenAIServiceError(
                "OpenAI embeddings API is currently unavailable."
            ) from exc
        self._validate(embeddings)
        logger.info("Generated %d embeddings with %s", len(embeddings), self.model)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]
