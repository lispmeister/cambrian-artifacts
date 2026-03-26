"""Tests for the test artifact server."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient


def make_app() -> web.Application:
    from server import handle_health, handle_stats
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


@pytest.fixture
async def client(aiohttp_client: pytest.fixture) -> TestClient:
    return await aiohttp_client(make_app())


@pytest.mark.asyncio
async def test_health_returns_ok(client: TestClient) -> None:
    resp = await client.get("/health")
    assert resp.status == 200
    data = await resp.json()
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_stats_returns_generation(client: TestClient) -> None:
    resp = await client.get("/stats")
    assert resp.status == 200
    data = await resp.json()
    assert "generation" in data
    assert "status" in data
    assert "uptime" in data
