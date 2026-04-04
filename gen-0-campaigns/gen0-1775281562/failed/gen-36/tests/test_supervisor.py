"""Tests for the Supervisor API client."""
from __future__ import annotations

import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions() returns list of records."""
    from src.supervisor import SupervisorClient

    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_stats() returns stats dict."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 5, "status": "idle", "uptime": 100}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn() returns ok response."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 1}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=1,
        artifact_path="gen-1",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.promote() sends correct request."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.update(body)
        return web.json_response({"ok": True, "generation": body.get("generation")})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert received["generation"] == 3


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.rollback() sends correct request."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.update(body)
        return web.json_response({"ok": True, "generation": body.get("generation")})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(4)
    assert result["ok"] is True
    assert received["generation"] == 4


async def test_supervisor_returns_empty_list_on_invalid(
    aiohttp_server: pytest.fixture,
) -> None:
    """get_versions() returns empty list when server returns non-list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn() sends spec-hash, generation, artifact-path in body."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.update(body)
        return web.json_response({"ok": True, "generation": body.get("generation")})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    spec_hash = "sha256:" + "b" * 64
    await client.spawn(
        spec_hash=spec_hash,
        generation=7,
        artifact_path="gen-7",
    )
    assert received["spec-hash"] == spec_hash
    assert received["generation"] == 7
    assert received["artifact-path"] == "gen-7"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions() returns empty list when server returns empty array."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
