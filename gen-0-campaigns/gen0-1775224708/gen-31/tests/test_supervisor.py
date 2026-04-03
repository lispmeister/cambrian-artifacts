"""Tests for the Supervisor API client."""
from __future__ import annotations

import json

from aiohttp import web


async def test_get_versions(aiohttp_server) -> None:
    """get_versions returns list of generation records."""
    from src.supervisor import SupervisorClient

    records = [{"generation": 1, "outcome": "promoted"}]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server) -> None:
    """get_stats returns stats dict."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 1, "status": "idle", "uptime": 100}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server) -> None:
    """spawn sends correct request and returns response."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 2}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:abc",
        generation=2,
        artifact_path="gen-2",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server) -> None:
    """promote sends correct request."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(generation=3)
    assert result["ok"] is True
    assert received[0]["generation"] == 3


async def test_rollback(aiohttp_server) -> None:
    """rollback sends correct request."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(generation=3)
    assert result["ok"] is True
    assert received[0]["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server) -> None:
    """get_versions returns empty list when response is not a list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server) -> None:
    """spawn sends spec-hash, generation, and artifact-path fields."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "container-id": "c1", "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(
        spec_hash="sha256:abc123",
        generation=2,
        artifact_path="gen-2",
    )
    body = received[0]
    assert body["spec-hash"] == "sha256:abc123"
    assert body["generation"] == 2
    assert body["artifact-path"] == "gen-2"


async def test_get_versions_empty_history(aiohttp_server) -> None:
    """get_versions handles empty list response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
