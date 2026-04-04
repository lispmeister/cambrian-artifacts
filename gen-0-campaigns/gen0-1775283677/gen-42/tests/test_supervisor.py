"""Tests for Supervisor API client."""
from __future__ import annotations

from aiohttp import web


async def test_get_versions(aiohttp_server) -> None:
    """SupervisorClient.get_versions returns list from server."""
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


async def test_get_stats(aiohttp_server) -> None:
    """SupervisorClient.get_stats returns stats from server."""
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


async def test_spawn(aiohttp_server) -> None:
    """SupervisorClient.spawn posts to /spawn and returns response."""
    from src.supervisor import SupervisorClient

    response_data = {"ok": True, "container-id": "test-container", "generation": 2}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=2,
        artifact_path="gen-2",
    )
    assert result == response_data


async def test_promote(aiohttp_server) -> None:
    """SupervisorClient.promote posts to /promote."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(5)
    assert result["ok"] is True
    assert received[0]["generation"] == 5


async def test_rollback(aiohttp_server) -> None:
    """SupervisorClient.rollback posts to /rollback."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(3)
    assert result["ok"] is True
    assert received[0]["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server) -> None:
    """get_versions returns empty list when server returns non-list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server) -> None:
    """spawn sends spec-hash, generation, artifact-path fields."""
    from src.supervisor import SupervisorClient

    received: list[dict] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    spec_hash = "sha256:" + "b" * 64
    client = SupervisorClient(str(server.make_url("/")))
    await client.spawn(spec_hash=spec_hash, generation=3, artifact_path="gen-3")

    assert len(received) == 1
    body = received[0]
    assert body["spec-hash"] == spec_hash
    assert body["generation"] == 3
    assert body["artifact-path"] == "gen-3"


async def test_get_versions_empty_history(aiohttp_server) -> None:
    """get_versions returns empty list when server returns []."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []
