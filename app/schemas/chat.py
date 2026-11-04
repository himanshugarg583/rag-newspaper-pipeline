"""Chat API schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.chat_message import MessageRole


class ChatRequest(BaseModel):
    session_id: uuid.UUID | None = None
    question: str = Field(min_length=1, max_length=4000)


class SourceCitation(BaseModel):
    document_id: uuid.UUID
    filename: str
    page_number: int
    chunk_id: uuid.UUID
    relevance_distance: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    question: str
    answer: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    sources: list[dict] | None
    created_at: datetime


class ChatHistoryResponse(BaseModel):
    session_id: uuid.UUID
    messages: list[ChatMessageResponse]
