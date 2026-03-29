"""Tests for the Supervisor API client."""
from __future__ import annotations

import json
from typing import Any

import pytest
from aiohttp import web

from src.supervisor import SupervisorClient


async def test_get_versions(aiohttp_server: Any) -> None:
    """get_versions returns list of generation records."""
    records = [
        {"generation": 1, "parent": 0, "outcome": "promoted"},
        {"generation": 2, "parent": 1, "outcome": "in_progress"},
    ]

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(records)

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == records


async def test_get_stats(aiohttp_server: Any) -> None:
    """get_stats returns stats dict."""
    stats = {"generation": 2, "status": "idle", "uptime": 120}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(stats)

    app = web.Application()
    app.router.add_get("/stats", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_stats()
    assert result == stats


async def test_spawn(aiohttp_server: Any) -> None:
    """spawn returns ok response."""
    response_data = {"ok": True, "container-id": "test-container", "generation": 3}

    async def handler(request: web.Request) -> web.Response:
        return web.json_response(response_data)

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.spawn(
        spec_hash="sha256:" + "a" * 64,
        generation=3,
        artifact_path="gen-3",
    )
    assert result["ok"] is True


async def test_promote(aiohttp_server: Any) -> None:
    """promote sends correct generation and returns ok."""
    received: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": received.get("generation")})

    app = web.Application()
    app.router.add_post("/promote", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.promote(5)
    assert result["ok"] is True
    assert received["generation"] == 5


async def test_rollback(aiohttp_server: Any) -> None:
    """rollback sends correct generation and returns ok."""
    received: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "generation": received.get("generation")})

    app = web.Application()
    app.router.add_post("/rollback", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.rollback(3)
    assert result["ok"] is True
    assert received["generation"] == 3


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: Any) -> None:
    """get_versions returns empty list when server returns non-list."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"error": "not a list"})

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []


async def test_spawn_sends_correct_fields(aiohttp_server: Any) -> None:
    """spawn sends spec-hash, generation, and artifact-path."""
    received: dict[str, Any] = {}

    async def handler(request: web.Request) -> web.Response:
        received.update(await request.json())
        return web.json_response({"ok": True, "container-id": "c1", "generation": 1})

    app = web.Application()
    app.router.add_post("/spawn", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    spec_hash = "sha256:" + "b" * 64
    await client.spawn(spec_hash=spec_hash, generation=1, artifact_path="gen-1")

    assert received["spec-hash"] == spec_hash
    assert received["generation"] == 1
    assert received["artifact-path"] == "gen-1"


async def test_get_versions_empty_history(aiohttp_server: Any) -> None:
    """get_versions handles empty list response."""
    async def handler(request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application()
    app.router.add_get("/versions", handler)
    server = await aiohttp_server(app)

    client = SupervisorClient(str(server.make_url("/")))
    result = await client.get_versions()
    assert result == []