"""Tests for Supervisor API client using aiohttp_server fixture."""

from __future__ import annotations

from typing import Any

from aiohttp import web

from src.supervisor import SupervisorClient


# ---------------------------------------------------------------------------
# Mock handlers
# ---------------------------------------------------------------------------


async def versions_handler(request: web.Request) -> web.Response:
    data = [
        {
            "generation": 1,
            "parent": 0,
            "spec-hash": "sha256:" + "a" * 64,
            "artifact-hash": "sha256:" + "b" * 64,
            "outcome": "promoted",
            "created": "2026-01-01T00:00:00Z",
            "container-id": "lab-gen-1",
        }
    ]
    return web.json_response(data)


async def stats_handler(request: web.Request) -> web.Response:
    return web.json_response({"generation": 1, "status": "idle", "uptime": 42})


async def spawn_handler(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response(
        {"ok": True, "container-id": "lab-gen-2", "generation": body["generation"]}
    )


async def promote_handler(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response({"ok": True, "generation": body["generation"]})


async def rollback_handler(request: web.Request) -> web.Response:
    body = await request.json()
    return web.json_response({"ok": True, "generation": body["generation"]})


def make_supervisor_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/versions", versions_handler)
    app.router.add_get("/stats", stats_handler)
    app.router.add_post("/spawn", spawn_handler)
    app.router.add_post("/promote", promote_handler)
    app.router.add_post("/rollback", rollback_handler)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_get_versions(aiohttp_server: Any) -> None:
    server = await aiohttp_server(make_supervisor_app())
    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert isinstance(versions, list)
    assert len(versions) == 1
    assert versions[0]["generation"] == 1
    assert versions[0]["outcome"] == "promoted"


async def test_get_stats(aiohttp_server: Any) -> None:
    server = await aiohttp_server(make_supervisor_app())
    client = SupervisorClient(str(server.make_url("")))
    stats = await client.get_stats()
    assert stats["generation"] == 1
    assert stats["status"] == "idle"
    assert stats["uptime"] == 42


async def test_spawn(aiohttp_server: Any) -> None:
    server = await aiohttp_server(make_supervisor_app())
    client = SupervisorClient(str(server.make_url("")))
    resp = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=2,
        artifact_path="gen-2",
    )
    assert resp["ok"] is True
    assert resp["generation"] == 2
    assert "container-id" in resp


async def test_promote(aiohttp_server: Any) -> None:
    server = await aiohttp_server(make_supervisor_app())
    client = SupervisorClient(str(server.make_url("")))
    resp = await client.promote(2)
    assert resp["ok"] is True
    assert resp["generation"] == 2


async def test_rollback(aiohttp_server: Any) -> None:
    server = await aiohttp_server(make_supervisor_app())
    client = SupervisorClient(str(server.make_url("")))
    resp = await client.rollback(2)
    assert resp["ok"] is True
    assert resp["generation"] == 2


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: Any) -> None:
    """Client handles unexpected response gracefully."""

    async def bad_versions(request: web.Request) -> web.Response:
        return web.json_response({"not": "a list"})

    app = web.Application()
    app.router.add_get("/versions", bad_versions)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert versions == []


async def test_spawn_sends_correct_fields(aiohttp_server: Any) -> None:
    """Spawn request should include all required fields."""
    received: dict[str, Any] = {}

    async def capture_spawn(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "container-id": "test", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", capture_spawn)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("")))

    await client.spawn(
        spec_hash="sha256:" + "f" * 64,
        generation=5,
        artifact_path="gen-5",
    )

    assert received["spec-hash"] == "sha256:" + "f" * 64
    assert received["generation"] == 5
    assert received["artifact-path"] == "gen-5"


async def test_get_versions_empty_history(aiohttp_server: Any) -> None:
    """Empty version list is handled correctly."""

    async def empty_versions(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", empty_versions)
    server = await aiohttp_server(app)
    client = SupervisorClient(str(server.make_url("")))
    versions = await client.get_versions()
    assert versions == []