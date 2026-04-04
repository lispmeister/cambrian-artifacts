"""Tests for the Supervisor API client."""

from __future__ import annotations

import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions returns list of records."""
    from src.supervisor import SupervisorClient

    records = [{"generation": 1, "outcome": "promoted"}]

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_stats returns stats dict."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 1, "status": "idle", "uptime": 100}

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handle_stats)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn posts correct data."""
    from src.supervisor import SupervisorClient

    async def handle_spawn(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "container-id": "test-container", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=1,
        artifact_path="gen-1",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.promote posts correct data."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handle_promote(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": received.get("generation")})

    app = web.Application()
    app.router.add_post("/promote", handle_promote)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(5)
    assert result["ok"] is True
    assert received["generation"] == 5


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.rollback posts correct data."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handle_rollback(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": received.get("generation")})

    app = web.Application()
    app.router.add_post("/rollback", handle_rollback)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(3)
    assert result["ok"] is True
    assert received["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list on non-list response."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response({"error": "no versions"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn sends spec-hash, generation, artifact-path."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handle_spawn(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "container-id": "c", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    spec_hash = "sha256:" + "b" * 64
    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(
        spec_hash=spec_hash,
        generation=2,
        artifact_path="gen-2",
    )
    assert received["spec-hash"] == spec_hash
    assert received["generation"] == 2
    assert received["artifact-path"] == "gen-2"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list when supervisor has no history."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []
