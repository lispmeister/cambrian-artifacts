"""Tests for Supervisor API client."""
import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns list of records."""
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
    """get_stats returns stats dict."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 5, "status": "idle", "uptime": 100}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """spawn sends correct request and returns response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
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


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """promote sends correct request."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(1)
    assert result["ok"] is True


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """rollback sends correct request."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(1)
    assert result["ok"] is True


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list on non-list response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn sends spec-hash, generation, and artifact-path."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "c1", "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    spec_hash = "sha256:" + "b" * 64
    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(spec_hash=spec_hash, generation=2, artifact_path="gen-2")

    assert received_body["spec-hash"] == spec_hash
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

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []
