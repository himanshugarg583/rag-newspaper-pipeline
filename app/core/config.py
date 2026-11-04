"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Newspaper RAG API", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")

    # CORS
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/newspaper_rag",
        alias="DATABASE_URL",
    )

    # Redis / Celery
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", alias="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", alias="CELERY_RESULT_BACKEND"
    )

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_embedding_dimensions: int = Field(
        default=1536, alias="OPENAI_EMBEDDING_DIMENSIONS"
    )
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")
    openai_request_timeout: float = Field(default=60.0, alias="OPENAI_REQUEST_TIMEOUT")
    openai_max_retries: int = Field(default=3, alias="OPENAI_MAX_RETRIES")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    # Chunking
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Retrieval
    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    hnsw_ef_search: int = Field(default=100, alias="HNSW_EF_SEARCH")
    # Optional cosine-distance cutoff (0 disables it). Calibrate against real
    # data before enabling: cosine distance of text-embedding-3-small pairs for
    # relevant matches usually falls well below 0.6.
    retrieval_max_distance: float = Field(default=0.0, alias="RETRIEVAL_MAX_DISTANCE")

    # Chat history
    chat_history_max_messages: int = Field(
        default=10, alias="CHAT_HISTORY_MAX_MESSAGES"
    )

    # Upload
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    upload_dir: str = Field(default="storage/newspapers", alias="UPLOAD_DIR")

    # PDF extraction / OCR
    min_extractable_chars: int = Field(default=200, alias="MIN_EXTRACTABLE_CHARS")
    enable_ocr: bool = Field(default=True, alias="ENABLE_OCR")
    ocr_languages: str = Field(default="eng+hin", alias="OCR_LANGUAGES")
    ocr_dpi: int = Field(default=200, alias="OCR_DPI")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def sync_database_url(self) -> str:
        """Synchronous driver URL for Alembic and Celery workers."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
