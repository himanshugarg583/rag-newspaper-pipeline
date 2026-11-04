"""Redis connection helpers."""

import redis.asyncio as aioredis

from app.core.config import settings

_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Return a shared async Redis client (lazy singleton)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def check_redis_health() -> bool:
    """Ping Redis; return True when reachable."""
    try:
        client = get_redis_client()
        return bool(await client.ping())
    except Exception:
        return False


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
