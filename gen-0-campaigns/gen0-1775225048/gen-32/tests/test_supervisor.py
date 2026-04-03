"""Tests for the Supervisor API client."""
from __future__ import annotations

import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server) -> None:
    """SupervisorClient.get_versions() returns list of records."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([
            {"generation": 1, "outcome": "promoted"},
            {"generation": 2, "outcome": "failed"},
        ])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert len(result) == 2
    assert result[0]["generation"] == 1


async def test_get_stats(aiohttp_server) -> None:
    """SupervisorClient.get_stats() returns stats dict."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"generation": 5, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result["generation"] == 5
    assert result["status"] == "idle"


async def test_spawn(aiohttp_server) -> None:
    """SupervisorClient.spawn() sends correct request and returns response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "container-id": "lab-gen-1", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:abc123",
        generation=1,
        artifact_path="gen-1",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server) -> None:
    """SupervisorClient.promote() sends correct request."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(1)
    assert result["ok"] is True
    assert received_body["generation"] == 1


async def test_rollback(aiohttp_server) -> None:
    """SupervisorClient.rollback() sends correct request."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "generation": 2})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(2)
    assert result["ok"] is True
    assert received_body["generation"] == 2


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server) -> None:
    """get_versions() returns empty list on non-200 response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=500, text="Internal Server Error")

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server) -> None:
    """spawn() sends spec-hash, generation, and artifact-path in request body."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "c1", "generation": 3})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(
        spec_hash="sha256:deadbeef",
        generation=3,
        artifact_path="gen-3",
    )
    assert received_body["spec-hash"] == "sha256:deadbeef"
    assert received_body["generation"] == 3
    assert received_body["artifact-path"] == "gen-3"


async def test_get_versions_empty_history(aiohttp_server) -> None:
    """get_versions() returns empty list when no generations exist."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
