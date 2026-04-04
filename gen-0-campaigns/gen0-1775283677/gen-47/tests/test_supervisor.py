"""Tests for the Supervisor API client."""
import pytest
from aiohttp import web


async def test_get_versions(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns list of generation records."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    versions = await client.get_versions()
    assert isinstance(versions, list)
    assert len(versions) == 1
    assert versions[0]["generation"] == 1


async def test_get_stats(aiohttp_server: pytest.fixture) -> None:
    """get_stats returns stats dict."""
    from src.supervisor import SupervisorClient

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response({"generation": 3, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handle_stats)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    stats = await client.get_stats()
    assert stats["generation"] == 3
    assert stats["status"] == "idle"


async def test_spawn(aiohttp_server: pytest.fixture) -> None:
    """spawn posts correct data and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_spawn(request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "container-id": "test-container", "generation": 2})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn("sha256:" + "a" * 64, 2, "gen-2")
    assert result["ok"] is True
    assert result["container-id"] == "test-container"


async def test_promote(aiohttp_server: pytest.fixture) -> None:
    """promote posts generation and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_promote(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handle_promote)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(3)
    assert result["ok"] is True
    assert result["generation"] == 3


async def test_rollback(aiohttp_server: pytest.fixture) -> None:
    """rollback posts generation and returns response."""
    from src.supervisor import SupervisorClient

    async def handle_rollback(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handle_rollback)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(3)
    assert result["ok"] is True
    assert result["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list on non-list response."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    versions = await client.get_versions()
    assert versions == []


async def test_spawn_sends_correct_fields(aiohttp_server: pytest.fixture) -> None:
    """spawn sends spec-hash, generation, artifact-path fields."""
    from src.supervisor import SupervisorClient

    received_body: dict = {}

    async def handle_spawn(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "c1", "generation": 5})

    app = web.Application()
    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    spec_hash = "sha256:" + "b" * 64
    await client.spawn(spec_hash, 5, "gen-5")
    assert received_body["spec-hash"] == spec_hash
    assert received_body["generation"] == 5
    assert received_body["artifact-path"] == "gen-5"


async def test_get_versions_empty_history(aiohttp_server: pytest.fixture) -> None:
    """get_versions returns empty list when history is empty."""
    from src.supervisor import SupervisorClient

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    versions = await client.get_versions()
    assert versions == []
