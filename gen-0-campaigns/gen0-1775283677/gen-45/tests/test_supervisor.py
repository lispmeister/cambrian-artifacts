"""Tests for Supervisor API client."""
import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions returns list from /versions."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert len(versions) == 1
    assert versions[0]["generation"] == 1


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_stats returns stats from /stats."""
    from src.supervisor import SupervisorClient

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response({"generation": 5, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handle_stats)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    stats = await client.get_stats()
    assert stats["generation"] == 5
    assert stats["status"] == "idle"


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn posts to /spawn and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_spawn(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "container-id": "test-container", "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=2,
        artifact_path="gen-2",
    )
    assert result["ok"] is True
    assert result["generation"] == 2


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.promote posts to /promote and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_promote(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 3})

    app = web.Application()
    app.router.add_post("/promote", handle_promote)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert result["generation"] == 3


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.rollback posts to /rollback and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_rollback(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "generation": 2})

    app = web.Application()
    app.router.add_post("/rollback", handle_rollback)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(2)
    assert result["ok"] is True


async def test_supervisor_returns_empty_list_on_invalid(
    aiohttp_server: pytest.fixture,
) -> None:
    """SupervisorClient.get_versions returns [] on non-200 response."""
    from src.supervisor import SupervisorClient

    async def handle_error(request: web.Request) -> web.Response:
        return web.Response(status=500, text="Internal Server Error")

    app = web.Application()
    app.router.add_get("/versions", handle_error)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn sends spec-hash, generation, artifact-path."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handle_spawn(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(
        spec_hash="sha256:" + "b" * 64,
        generation=1,
        artifact_path="gen-1",
    )
    assert "spec-hash" in received_body
    assert "generation" in received_body
    assert "artifact-path" in received_body
    assert received_body["generation"] == 1
    assert received_body["artifact-path"] == "gen-1"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions returns empty list when no history."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert versions == []
