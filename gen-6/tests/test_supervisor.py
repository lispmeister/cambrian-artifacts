"""Tests for Supervisor API client using aiohttp_server fixture."""

import pytest
from aiohttp import web

from src.supervisor import SupervisorClient


async def test_get_versions(aiohttp_server) -> None:
    """get_versions returns list from /versions endpoint."""
    records = [
        {
            "generation": 1,
            "parent": 0,
            "spec-hash": "sha256:abc",
            "artifact-hash": "sha256:def",
            "outcome": "promoted",
            "created": "2026-01-01T00:00:00Z",
            "container-id": "test-1",
        }
    ]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["generation"] == 1


async def test_get_stats(aiohttp_server) -> None:
    """get_stats returns dict from /stats endpoint."""
    stats = {"generation": 1, "status": "idle", "uptime": 100}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result["generation"] == 1
    assert result["status"] == "idle"


async def test_spawn(aiohttp_server) -> None:
    """spawn sends correct request and returns response."""
    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "container-id": "test-c", "generation": 5})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:abc",
        generation=5,
        artifact_path="gen-5",
    )
    assert result["ok"] is True
    assert received["generation"] == 5
    assert received["spec-hash"] == "sha256:abc"
    assert received["artifact-path"] == "gen-5"


async def test_promote(aiohttp_server) -> None:
    """promote sends correct request."""
    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": 3})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert received["generation"] == 3


async def test_rollback(aiohttp_server) -> None:
    """rollback sends correct request."""
    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": 4})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(4)
    assert result["ok"] is True
    assert received["generation"] == 4


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server) -> None:
    """get_versions returns empty list on non-200 response."""
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="error")

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server) -> None:
    """spawn request body contains spec-hash, generation, artifact-path."""
    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "container-id": "c1", "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(
        spec_hash="sha256:deadbeef",
        generation=2,
        artifact_path="gen-2",
    )
    assert "spec-hash" in received
    assert "generation" in received
    assert "artifact-path" in received


async def test_get_versions_empty_history(aiohttp_server) -> None:
    """get_versions handles empty array response."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []