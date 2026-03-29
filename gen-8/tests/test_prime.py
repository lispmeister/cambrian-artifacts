"""Tests for Prime HTTP API (/health, /stats)."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient

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
    """GET /health returns application/json."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert "application/json" in resp.content_type


async def test_stats_status(aiohttp_client: Any) -> None:
    """GET /stats returns 200."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client: Any) -> None:
    """GET /stats has generation, status, uptime keys."""
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
    """GET /stats status is one of idle, generating, verifying."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["status"] in ("idle", "generating", "verifying")


async def test_stats_generation_default(aiohttp_client: Any) -> None:
    """GET /stats generation defaults to 0 when CAMBRIAN_GENERATION not set."""
    import os
    os.environ.pop("CAMBRIAN_GENERATION", None)
    # Re-import to get fresh value
    import importlib
    import src.prime as prime_module
    importlib.reload(prime_module)
    client = await aiohttp_client(prime_module.make_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["generation"] == 0


async def test_unknown_route(aiohttp_client: Any) -> None:
    """Unknown route returns 404."""
    client = await aiohttp_client(make_app())
    resp = await client.get("/unknown")
    assert resp.status == 404


# Type annotation for aiohttp_client fixture
from typing import Any