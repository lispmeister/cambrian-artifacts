"""Tests for Supervisor client using aiohttp_server fixture."""
from __future__ import annotations

from aiohttp import web


async def test_get_versions(aiohttp_server) -> None:
    """GET /versions returns list of generation records."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([{"generation": 1, "outcome": "promoted"}])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        versions = await client.get_versions()
        assert len(versions) == 1
        assert versions[0]["generation"] == 1
    finally:
        await client.close()


async def test_get_stats(aiohttp_server) -> None:
    """GET /stats returns supervisor stats."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"generation": 5, "status": "idle", "uptime": 100})

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        stats = await client.get_stats()
        assert stats["generation"] == 5
        assert stats["status"] == "idle"
    finally:
        await client.close()


async def test_spawn(aiohttp_server) -> None:
    """POST /spawn returns ok response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "container-id": "test-1", "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        result = await client.spawn("sha256:" + "a" * 64, 1, "gen-1")
        assert result["ok"] is True
    finally:
        await client.close()


async def test_promote(aiohttp_server) -> None:
    """POST /promote returns ok response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        result = await client.promote(1)
        assert result["ok"] is True
    finally:
        await client.close()


async def test_rollback(aiohttp_server) -> None:
    """POST /rollback returns ok response."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body["generation"]})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        result = await client.rollback(1)
        assert result["ok"] is True
    finally:
        await client.close()


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server) -> None:
    """GET /versions with non-list response returns empty list."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        versions = await client.get_versions()
        assert versions == []
    finally:
        await client.close()


async def test_spawn_sends_correct_fields(aiohttp_server) -> None:
    """POST /spawn sends spec-hash, generation, artifact-path."""
    from src.supervisor import SupervisorClient
    received_body = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "container-id": "t", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        await client.spawn("sha256:" + "b" * 64, 42, "gen-42")
        assert received_body["spec-hash"] == "sha256:" + "b" * 64
        assert received_body["generation"] == 42
        assert received_body["artifact-path"] == "gen-42"
    finally:
        await client.close()


async def test_get_versions_empty_history(aiohttp_server) -> None:
    """GET /versions with empty list works."""
    from src.supervisor import SupervisorClient

    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    try:
        versions = await client.get_versions()
        assert versions == []
    finally:
        await client.close()
