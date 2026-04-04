"""Tests for the Supervisor API client."""

import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns list of generation records."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["generation"] == 1


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """get_stats returns supervisor stats."""
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


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """spawn sends POST /spawn and returns response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "container-id": "test-container", "generation": 1})

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
    """promote sends POST /promote."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 2})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(2)
    assert result["ok"] is True


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """rollback sends POST /rollback."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 3})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(3)
    assert result["ok"] is True


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns [] when server returns non-list."""
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
    """spawn sends correct fields in request body."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "c1", "generation": 5})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    spec_hash = "sha256:" + "b" * 64
    await client.spawn(spec_hash=spec_hash, generation=5, artifact_path="gen-5")
    assert received_body["spec-hash"] == spec_hash
    assert received_body["generation"] == 5
    assert received_body["artifact-path"] == "gen-5"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns [] when no history."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
