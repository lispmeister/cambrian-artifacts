"""Tests for Prime HTTP API (/health, /stats)."""

import pytest
from aiohttp import web


async def test_health_status(aiohttp_client) -> None:
    """GET /health returns 200."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_body(aiohttp_client) -> None:
    """GET /health returns {"ok": true}."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    data = await resp.json()
    assert data == {"ok": True}


async def test_health_content_type(aiohttp_client) -> None:
    """GET /health returns JSON content type."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert "application/json" in resp.content_type


async def test_stats_status(aiohttp_client) -> None:
    """GET /stats returns 200."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client) -> None:
    """GET /stats response has generation, status, uptime keys."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert "generation" in data
    assert "status" in data
    assert "uptime" in data


async def test_stats_generation_type(aiohttp_client) -> None:
    """GET /stats generation is an integer."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["generation"], int)


async def test_stats_uptime_type(aiohttp_client) -> None:
    """GET /stats uptime is an integer."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["uptime"], int)


async def test_stats_status_value(aiohttp_client) -> None:
    """GET /stats status is one of the valid values."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["status"] in ("idle", "generating", "verifying")


async def test_stats_generation_default(aiohttp_client) -> None:
    """GET /stats generation defaults to 0 when CAMBRIAN_GENERATION not set."""
    import os
    import src.prime as prime_module

    original = prime_module._generation
    prime_module._generation = 0
    try:
        from src.prime import make_app
        client = await aiohttp_client(make_app())
        resp = await client.get("/stats")
        data = await resp.json()
        assert data["generation"] == 0
    finally:
        prime_module._generation = original


async def test_unknown_route(aiohttp_client) -> None:
    """Unknown route returns 404."""
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/unknown")
    assert resp.status == 404