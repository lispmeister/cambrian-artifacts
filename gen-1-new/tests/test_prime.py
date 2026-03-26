"""Tests for Prime HTTP API (/health, /stats, /versions proxy)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web


@pytest.fixture
def make_client(aiohttp_client):
    """Return an async factory that creates a test client for the Prime app."""
    async def _make():
        from src.prime import make_app
        return await aiohttp_client(make_app())
    return _make


async def test_health_returns_200(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_body_ok_true(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    body = await resp.json()
    assert body["ok"] is True


async def test_stats_returns_200(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    body = await resp.json()
    assert "generation" in body
    assert "status" in body
    assert "uptime" in body


async def test_stats_generation_is_int(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    body = await resp.json()
    assert isinstance(body["generation"], int)


async def test_stats_generation_is_zero_before_loop(aiohttp_client):
    """Before the generation loop starts, /stats returns generation=0."""
    import src.prime as prime_module
    # Reset to 0 to simulate fresh start
    original = prime_module._generation
    prime_module._generation = 0
    try:
        client = await aiohttp_client(prime_module.make_app())
        resp = await client.get("/stats")
        body = await resp.json()
        assert body["generation"] == 0
    finally:
        prime_module._generation = original


async def test_stats_status_is_valid(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    body = await resp.json()
    assert body["status"] in ("idle", "generating", "verifying")


async def test_stats_uptime_is_non_negative(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    body = await resp.json()
    assert body["uptime"] >= 0


async def test_versions_proxies_to_supervisor(aiohttp_client):
    """GET /versions should return the array from supervisor.get_versions()."""
    from src.prime import make_app

    fake_records = [{"generation": 1, "outcome": "promoted"}]
    with patch("src.prime.supervisor.get_versions", new=AsyncMock(return_value=fake_records)):
        client = await aiohttp_client(make_app())
        resp = await client.get("/versions")
        assert resp.status == 200
        body = await resp.json()
        assert body == fake_records


async def test_versions_returns_empty_list_when_no_records(aiohttp_client):
    from src.prime import make_app

    with patch("src.prime.supervisor.get_versions", new=AsyncMock(return_value=[])):
        client = await aiohttp_client(make_app())
        resp = await client.get("/versions")
        assert resp.status == 200
        body = await resp.json()
        assert body == []
