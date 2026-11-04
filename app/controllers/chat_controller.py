"""Chat API endpoints."""

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service(db: AsyncSession = Depends(get_db)) -> ChatService:
    return ChatService(db)


@router.post("", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    session_id, answer, _sources = await service.ask(
        question=request.question, session_id=request.session_id
    )
    # Sources are stored internally with the chat message but intentionally
    # not returned in the chat response.
    return ChatResponse(
        session_id=session_id,
        question=request.question,
        answer=answer,
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: uuid.UUID,
    service: ChatService = Depends(get_chat_service),
) -> ChatHistoryResponse:
    messages = await service.get_history(session_id)
    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessageResponse.model_validate(m) for m in messages],
    )
