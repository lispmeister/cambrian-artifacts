"""Tests for the Prime HTTP API (/health and /stats)."""

from __future__ import annotations

import pytest


async def test_health_status(aiohttp_client: pytest.fixture) -> None:
    """GET /health returns 200."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_body(aiohttp_client: pytest.fixture) -> None:
    """GET /health returns {"ok": true}."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    data = await resp.json()
    assert data == {"ok": True}


async def test_health_content_type(aiohttp_client: pytest.fixture) -> None:
    """GET /health returns JSON content type."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert "application/json" in resp.content_type


async def test_stats_status(aiohttp_client: pytest.fixture) -> None:
    """GET /stats returns 200."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client: pytest.fixture) -> None:
    """GET /stats returns JSON with required keys."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert "generation" in data
    assert "status" in data
    assert "uptime" in data


async def test_stats_generation_type(aiohttp_client: pytest.fixture) -> None:
    """GET /stats generation field is an integer."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["generation"], int)


async def test_stats_uptime_type(aiohttp_client: pytest.fixture) -> None:
    """GET /stats uptime field is an integer."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["uptime"], int)


async def test_stats_status_value(aiohttp_client: pytest.fixture) -> None:
    """GET /stats status field is one of the valid values."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["status"] in ("idle", "generating", "verifying")


async def test_stats_generation_default(aiohttp_client: pytest.fixture) -> None:
    """GET /stats generation is 0 when CAMBRIAN_GENERATION not set."""
    import src.prime as prime_module
    from src.prime import make_app
    original = prime_module._prime_generation
    prime_module._prime_generation = 0
    try:
        client = await aiohttp_client(make_app())
        resp = await client.get("/stats")
        data = await resp.json()
        assert data["generation"] == 0
    finally:
        prime_module._prime_generation = original


async def test_unknown_route(aiohttp_client: pytest.fixture) -> None:
    """Unknown routes return 404."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/nonexistent")
    assert resp.status == 404
