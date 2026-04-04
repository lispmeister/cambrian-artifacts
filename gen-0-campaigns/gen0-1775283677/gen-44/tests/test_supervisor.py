"""Tests for Supervisor API client."""
from __future__ import annotations

import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns list of records."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["generation"] == 1


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """get_stats returns stats dict."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"generation": 5, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result["generation"] == 5
    assert result["status"] == "idle"


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """spawn sends correct request and returns response."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "container-id": "test-container", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=1,
        artifact_path="gen-1",
    )
    assert result["ok"] is True
    assert len(received) == 1


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """promote sends correct request."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(1)
    assert result["ok"] is True
    assert received[0]["generation"] == 1


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """rollback sends correct request."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": 2})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(2)
    assert result["ok"] is True
    assert received[0]["generation"] == 2


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list on unexpected response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"not": "a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn sends spec-hash, generation, artifact-path fields."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "container-id": "c1", "generation": 3})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    spec_hash = "sha256:" + "b" * 64
    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(spec_hash=spec_hash, generation=3, artifact_path="gen-3")

    assert received[0]["spec-hash"] == spec_hash
    assert received[0]["generation"] == 3
    assert received[0]["artifact-path"] == "gen-3"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions handles empty array response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []
