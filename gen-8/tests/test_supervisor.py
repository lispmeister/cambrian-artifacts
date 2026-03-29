"""Tests for Supervisor client (mock HTTP)."""

import json
from typing import Any

import pytest
from aiohttp import web

from src.supervisor import SupervisorClient


async def test_get_versions(aiohttp_server: Any) -> None:
    """get_versions returns list of records."""
    records = [{"generation": 1, "parent": 0, "outcome": "promoted"}]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: Any) -> None:
    """get_stats returns stats dict."""
    stats = {"generation": 1, "status": "idle", "uptime": 42}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: Any) -> None:
    """spawn posts correct body and returns response."""
    response_body = {"ok": True, "container-id": "lab-gen-7", "generation": 7}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["generation"] == 7
        assert "spec-hash" in body
        assert "artifact-path" in body
        return web.json_response(response_body)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=7,
        artifact_path="gen-7",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server: Any) -> None:
    """promote posts correct body."""
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["generation"] == 5
        return web.json_response({"ok": True, "generation": 5})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.promote(5)
    assert result["ok"] is True


async def test_rollback(aiohttp_server: Any) -> None:
    """rollback posts correct body."""
    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        assert body["generation"] == 3
        return web.json_response({"ok": True, "generation": 3})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.rollback(3)
    assert result["ok"] is True


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: Any) -> None:
    """get_versions returns [] when server returns non-list."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "bad"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: Any) -> None:
    """spawn sends spec-hash, generation, artifact-path."""
    received_body: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        nonlocal received_body
        received_body = await request.json()
        return web.json_response({"ok": True, "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    await client.spawn(
        spec_hash="sha256:" + "b" * 64,
        generation=1,
        artifact_path="gen-1",
    )
    assert received_body["spec-hash"] == "sha256:" + "b" * 64
    assert received_body["generation"] == 1
    assert received_body["artifact-path"] == "gen-1"


async def test_get_versions_empty_history(aiohttp_server: Any) -> None:
    """get_versions returns [] on empty list."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("")))
    result = await client.get_versions()
    assert result == []