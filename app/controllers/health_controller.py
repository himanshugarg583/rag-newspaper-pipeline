"""Health check endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.redis import check_redis_health
from app.schemas.common import HealthStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthStatus:
    database_status = "healthy"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        database_status = "unhealthy"

    redis_status = "healthy" if await check_redis_health() else "unhealthy"

    overall = (
        "healthy"
        if database_status == "healthy" and redis_status == "healthy"
        else "degraded"
    )
    return HealthStatus(status=overall, database=database_status, redis=redis_status)
