"""Tests for Prime HTTP API endpoints."""

from __future__ import annotations

import os
from typing import Any

import pytest
from aiohttp import web


# ---------------------------------------------------------------------------
# App under test
# ---------------------------------------------------------------------------


def make_test_app() -> web.Application:
    """Create app for testing."""
    from src.prime import make_app
    return make_app()


# ---------------------------------------------------------------------------
# /health tests
# ---------------------------------------------------------------------------


async def test_health_status(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_body(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/health")
    data = await resp.json()
    assert data == {"ok": True}


async def test_health_content_type(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/health")
    assert "application/json" in resp.content_type


# ---------------------------------------------------------------------------
# /stats tests
# ---------------------------------------------------------------------------


async def test_stats_status(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/stats")
    assert resp.status == 200


async def test_stats_has_required_keys(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert "generation" in data
    assert "status" in data
    assert "uptime" in data


async def test_stats_generation_type(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["generation"], int)


async def test_stats_uptime_type(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert isinstance(data["uptime"], int)
    assert data["uptime"] >= 0


async def test_stats_status_value(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/stats")
    data = await resp.json()
    assert data["status"] in ("idle", "generating", "verifying")


async def test_stats_generation_default(aiohttp_client: Any) -> None:
    """When CAMBRIAN_GENERATION is not set, generation should be 0."""
    old_val = os.environ.pop("CAMBRIAN_GENERATION", None)
    try:
        import importlib
        import src.prime as prime_mod
        importlib.reload(prime_mod)
        client = await aiohttp_client(prime_mod.make_app())
        resp = await client.get("/stats")
        data = await resp.json()
        assert data["generation"] == 0
    finally:
        if old_val is not None:
            os.environ["CAMBRIAN_GENERATION"] = old_val


# ---------------------------------------------------------------------------
# 404 for unknown routes
# ---------------------------------------------------------------------------


async def test_unknown_route(aiohttp_client: Any) -> None:
    client = await aiohttp_client(make_test_app())
    resp = await client.get("/nonexistent")
    assert resp.status == 404