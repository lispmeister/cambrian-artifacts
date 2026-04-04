"""Tests for the Supervisor API client."""
import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions() returns a list of records."""
    from src.supervisor import SupervisorClient

    records = [{"generation": 1, "outcome": "promoted"}]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_stats() returns stats dict."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 1, "status": "idle", "uptime": 100}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn() sends correct request and returns response."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 2}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=2,
        artifact_path="gen-2",
    )
    assert result["ok"] is True
    assert result["container-id"] == "test-container"


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.promote() sends correct request."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.update(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
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
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(5)
    assert result["ok"] is True
    assert received["generation"] == 5


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions() returns empty list when response is not a list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn() sends the correct JSON fields."""
    from src.supervisor import SupervisorClient

    received: dict = {}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.update(body)
        return web.json_response({"ok": True, "container-id": "c1", "generation": 7})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(
        spec_hash="sha256:" + "b" * 64,
        generation=7,
        artifact_path="gen-7",
    )
    assert received["spec-hash"] == "sha256:" + "b" * 64
    assert received["generation"] == 7
    assert received["artifact-path"] == "gen-7"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions() handles empty history."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []
