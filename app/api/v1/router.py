"""API v1 router aggregating all controllers."""

from fastapi import APIRouter

from app.controllers.chat_controller import router as chat_router
from app.controllers.document_controller import router as document_router
from app.controllers.health_controller import router as health_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(document_router)
api_v1_router.include_router(chat_router)
