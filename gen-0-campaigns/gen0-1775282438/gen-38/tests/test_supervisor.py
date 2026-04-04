"""Tests for the Supervisor API client."""
import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.get_versions returns list from /versions."""
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
    """SupervisorClient.get_stats returns stats dict from /stats."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 5, "status": "idle", "uptime": 100}

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handle_stats)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.spawn posts to /spawn and returns result."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 2}

    async def handle_spawn(request: web.Request) -> web.Response:
        return web.json_response(response_data)

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
    """SupervisorClient.promote posts to /promote."""
    from src.supervisor import SupervisorClient

    received: list = []

    async def handle_promote(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handle_promote)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert received[0]["generation"] == 3


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """SupervisorClient.rollback posts to /rollback."""
    from src.supervisor import SupervisorClient

    received: list = []

    async def handle_rollback(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handle_rollback)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(3)
    assert result["ok"] is True
    assert received[0]["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(
    aiohttp_server: pytest.fixture,
) -> None:
    """get_versions returns empty list when response is not a list."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn sends spec-hash, generation, and artifact-path in body."""
    from src.supervisor import SupervisorClient

    received: list = []

    async def handle_spawn(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    spec_hash = "sha256:" + "c" * 64
    await client.spawn(
        spec_hash=spec_hash,
        generation=7,
        artifact_path="gen-7",
    )

    assert received[0]["spec-hash"] == spec_hash
    assert received[0]["generation"] == 7
    assert received[0]["artifact-path"] == "gen-7"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list when /versions returns []."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []
