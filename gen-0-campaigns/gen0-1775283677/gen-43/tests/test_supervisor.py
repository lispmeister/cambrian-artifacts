"""Tests for the Supervisor API client."""
from __future__ import annotations

import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
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


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """get_stats returns supervisor stats."""
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
    """spawn posts to /spawn and returns result."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 1}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:abc",
        generation=1,
        artifact_path="gen-1",
    )
    assert result == response_data


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """promote posts to /promote and returns result."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "generation": 1}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(1)
    assert result == response_data


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """rollback posts to /rollback and returns result."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "generation": 1}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(1)
    assert result == response_data


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list when response is not a list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "bad"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn sends correct JSON fields."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(
        spec_hash="sha256:testhash",
        generation=2,
        artifact_path="gen-2",
    )
    assert received_body["spec-hash"] == "sha256:testhash"
    assert received_body["generation"] == 2
    assert received_body["artifact-path"] == "gen-2"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list when history is empty."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
