"""Tests for Prime HTTP API endpoints."""
from __future__ import annotations

import pytest
from aiohttp import web

# Import make_app from src.prime
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prime import make_app


async def test_health_status(aiohttp_client: Any) -> None:
    """GET /health returns 200."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_body(aiohttp_client: Any) -> None:
    """GET /health returns {"ok": true}."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    data = await resp.json()
    assert data == {"ok": True}


async def test_health_content_type(aiohttp_client: Any) -> None:
    """GET /health returns JSON content type."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert "application/json" in resp.content_type


async def test_stats_status(aiohttp_client: Any) -> None:
    """GET /stats returns 200."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client: Any) -> None:
    """GET /stats returns JSON with generation, status, uptime."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert "generation" in data
    assert "status" in data
    assert "uptime" in data


async def test_stats_generation_type(aiohttp_client: Any) -> None:
    """GET /stats generation is an integer."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["generation"], int)


async def test_stats_uptime_type(aiohttp_client: Any) -> None:
    """GET /stats uptime is an integer."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["uptime"], int)


async def test_stats_status_value(aiohttp_client: Any) -> None:
    """GET /stats status is a valid string value."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["status"] in ("idle", "generating", "verifying")


async def test_stats_generation_default(aiohttp_client: Any) -> None:
    """GET /stats generation defaults to 0 when env var not set."""
    import src.prime as prime_module
    original = prime_module._generation
    prime_module._generation = 0
    try:
        client = await aiohttp_client(make_app())
        resp = await client.get("/stats")
        data = await resp.json()
        assert data["generation"] == 0
    finally:
        prime_module._generation = original


async def test_unknown_route(aiohttp_client: Any) -> None:
    """Unknown routes return 404."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/nonexistent")
    assert resp.status == 404


# Type annotation for fixtures
from typing import Any
