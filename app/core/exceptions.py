"""Application exceptions and centralized FastAPI exception handlers."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


class AppException(Exception):
    """Base class for domain exceptions mapped to HTTP responses."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str = "An internal error occurred."):
        self.message = message
        super().__init__(message)


class InvalidFileTypeError(AppException):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "INVALID_FILE_TYPE"


class FileSizeExceededError(AppException):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    error_code = "FILE_SIZE_EXCEEDED"


class DuplicateDocumentError(AppException):
    status_code = status.HTTP_409_CONFLICT
    error_code = "DUPLICATE_DOCUMENT"


class InvalidPDFError(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "INVALID_PDF"


class ScannedPDFError(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "SCANNED_OR_EMPTY_PDF"


class CorruptedPDFTextError(AppException):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code = "CORRUPTED_OR_LEGACY_FONT_PDF"


class DocumentNotFoundError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "DOCUMENT_NOT_FOUND"


class DocumentNotReprocessableError(AppException):
    status_code = status.HTTP_409_CONFLICT
    error_code = "DOCUMENT_NOT_REPROCESSABLE"


class ChatSessionNotFoundError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "CHAT_SESSION_NOT_FOUND"


class NoRelevantContextError(AppException):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "NO_RELEVANT_CONTEXT"


class EmbeddingDimensionError(AppException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "EMBEDDING_DIMENSION_MISMATCH"


class OpenAIServiceError(AppException):
    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "OPENAI_SERVICE_ERROR"


def _error_response(status_code: int, error_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {"code": error_code, "message": message},
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
        logger.warning(
            "Application error: %s (%s) on %s %s",
            exc.message,
            exc.error_code,
            request.method,
            request.url.path,
        )
        return _error_response(exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed.",
                    "details": exc.errors(),
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Never leak internal exception details to clients.
        logger.exception(
            "Unhandled error on %s %s", request.method, request.url.path
        )
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "An internal error occurred.",
        )
