"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.redis import close_redis
from app.utils.file_utils import ensure_directory

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directory(settings.upload_dir)
    logger.info("%s started (env=%s)", settings.app_name, settings.app_env)
    yield
    await close_redis()
    logger.info("%s shut down", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)

# Serve the bundled single-page frontend at the root URL.
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount(
        "/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend"
    )
