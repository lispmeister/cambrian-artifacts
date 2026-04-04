"""Tests for Supervisor API client."""
import json
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """GET /versions returns list of generation records."""
    from src.supervisor import SupervisorClient

    records = [{"generation": 1, "outcome": "promoted"}]

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """GET /stats returns supervisor stats."""
    from src.supervisor import SupervisorClient

    stats = {"generation": 1, "status": "idle", "uptime": 100}

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handle_stats)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """POST /spawn sends correct request."""
    from src.supervisor import SupervisorClient

    async def handle_spawn(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "container-id": "test-container", "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=2,
        artifact_path="gen-2",
    )
    assert result.get("ok") is True


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """POST /promote sends correct request."""
    from src.supervisor import SupervisorClient

    async def handle_promote(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handle_promote)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(generation=2)
    assert result.get("ok") is True


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """POST /rollback sends correct request."""
    from src.supervisor import SupervisorClient

    async def handle_rollback(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handle_rollback)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(generation=2)
    assert result.get("ok") is True


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list on non-list response."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response({"error": "invalid"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """POST /spawn sends spec-hash, generation, and artifact-path."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handle_spawn(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "c", "generation": 3})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    spec_hash = "sha256:" + "b" * 64
    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(
        spec_hash=spec_hash,
        generation=3,
        artifact_path="gen-3",
    )
    assert received_body.get("spec-hash") == spec_hash
    assert received_body.get("generation") == 3
    assert received_body.get("artifact-path") == "gen-3"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """GET /versions with empty history returns empty list."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
