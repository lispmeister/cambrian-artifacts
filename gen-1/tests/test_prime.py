"""Tests for Prime HTTP API (/health, /stats)."""
import pytest


async def test_health_returns_200(aiohttp_client):
    from src.prime import make_app
    client = await aiohttp_client(make_app())
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_returns_ok(aiohttp_client):
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
