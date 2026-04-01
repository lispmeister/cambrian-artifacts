"""Tests for Supervisor API client."""
from __future__ import annotations

import json
import pytest
from aiohttp import web

from src.supervisor import SupervisorClient


async def test_get_versions(aiohttp_server: Any) -> None:
    """get_versions returns list of records."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert isinstance(versions, list)
    assert len(versions) == 1
    assert versions[0]["generation"] == 1


async def test_get_stats(aiohttp_server: Any) -> None:
    """get_stats returns dict."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"generation": 5, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    stats = await client.get_stats()
    assert stats["generation"] == 5
    assert stats["status"] == "idle"


async def test_spawn(aiohttp_server: Any) -> None:
    """spawn returns success response."""
    async def handler(request: web.Request) -> web.Response:
        data = await request.json()
        return web.json_response({"ok": True, "container-id": "test-container", "generation": data["generation"]})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:abc123",
        generation=2,
        artifact_path="gen-2",
    )
    assert result["ok"] is True
    assert result["generation"] == 2


async def test_promote(aiohttp_server: Any) -> None:
    """promote returns success response."""
    async def handler(request: web.Request) -> web.Response:
        data = await request.json()
        return web.json_response({"ok": True, "generation": data["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert result["generation"] == 3


async def test_rollback(aiohttp_server: Any) -> None:
    """rollback returns success response."""
    async def handler(request: web.Request) -> web.Response:
        data = await request.json()
        return web.json_response({"ok": True, "generation": data["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(3)
    assert result["ok"] is True


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: Any) -> None:
    """get_versions returns empty list on non-list response."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert versions == []


async def test_spawn_sends_correct_fields(aiohttp_server: Any) -> None:
    """spawn sends spec-hash, generation, artifact-path."""
    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": received["generation"]})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(
        spec_hash="sha256:deadbeef",
        generation=7,
        artifact_path="gen-7",
    )
    assert received["spec-hash"] == "sha256:deadbeef"
    assert received["generation"] == 7
    assert received["artifact-path"] == "gen-7"


async def test_get_versions_empty_history(aiohttp_server: Any) -> None:
    """get_versions returns empty list when history is empty."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert versions == []


# Type annotation for fixtures
from typing import Any
