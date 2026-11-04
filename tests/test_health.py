"""Health endpoint tests."""

from unittest.mock import AsyncMock, patch

async def test_health_endpoint_healthy(client, mock_db):
    mock_db.execute = AsyncMock()
    with patch(
        "app.controllers.health_controller.check_redis_health",
        new=AsyncMock(return_value=True),
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] == "healthy"
    assert body["redis"] == "healthy"


async def test_health_endpoint_degraded_when_redis_down(client, mock_db):
    mock_db.execute = AsyncMock()
    with patch(
        "app.controllers.health_controller.check_redis_health",
        new=AsyncMock(return_value=False),
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["redis"] == "unhealthy"
