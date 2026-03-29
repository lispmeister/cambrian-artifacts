"""Tests for Supervisor API client."""

from __future__ import annotations

import pytest
from aiohttp import web

from src.supervisor import SupervisorClient


async def make_supervisor_app(
    versions_response: list | None = None,
    stats_response: dict | None = None,
) -> web.Application:
    """Create a mock Supervisor application."""
    app = web.Application()

    versions_data = versions_response or []
    stats_data = stats_response or {"generation": 0, "status": "idle", "uptime": 0}

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response(versions_data)

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats_data)

    async def handle_spawn(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "container-id": "test-container", "generation": body.get("generation", 1)})

    async def handle_promote(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body.get("generation", 1)})

    async def handle_rollback(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"ok": True, "generation": body.get("generation", 1)})

    app.router.add_get("/versions", handle_versions)
    app.router.add_get("/stats", handle_stats)
    app.router.add_post("/spawn", handle_spawn)
    app.router.add_post("/promote", handle_promote)
    app.router.add_post("/rollback", handle_rollback)

    return app


async def test_get_versions(aiohttp_server: Any) -> None:
    """SupervisorClient.get_versions returns parsed records."""
    records = [
        {
            "generation": 1,
            "parent": 0,
            "spec-hash": "sha256:" + "a" * 64,
            "artifact-hash": "sha256:" + "b" * 64,
            "outcome": "promoted",
            "created": "2026-01-01T00:00:00Z",
            "container-id": "test-container",
        }
    ]
    app = await make_supervisor_app(versions_response=records)
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        versions = await client.get_versions()
        assert len(versions) == 1
        assert versions[0].generation == 1
        assert versions[0].outcome == "promoted"
    finally:
        await client.close()


async def test_get_stats(aiohttp_server: Any) -> None:
    """SupervisorClient.get_stats returns stats dict."""
    stats = {"generation": 3, "status": "idle", "uptime": 120}
    app = await make_supervisor_app(stats_response=stats)
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        result = await client.get_stats()
        assert result["generation"] == 3
        assert result["status"] == "idle"
    finally:
        await client.close()


async def test_spawn(aiohttp_server: Any) -> None:
    """SupervisorClient.spawn returns ok response."""
    app = await make_supervisor_app()
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        result = await client.spawn(
            spec_hash="sha256:" + "a" * 64,
            generation=1,
            artifact_path="gen-1",
        )
        assert result["ok"] is True
    finally:
        await client.close()


async def test_promote(aiohttp_server: Any) -> None:
    """SupervisorClient.promote returns ok response."""
    app = await make_supervisor_app()
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        result = await client.promote(generation=1)
        assert result["ok"] is True
    finally:
        await client.close()


async def test_rollback(aiohttp_server: Any) -> None:
    """SupervisorClient.rollback returns ok response."""
    app = await make_supervisor_app()
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        result = await client.rollback(generation=1)
        assert result["ok"] is True
    finally:
        await client.close()


async def test_supervisor_returns_empty_list_on_invalid(aiohttp_server: Any) -> None:
    """get_versions returns empty list on invalid records."""
    app = web.Application()

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response([{"invalid": "data"}])

    app.router.add_get("/versions", handle_versions)
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        versions = await client.get_versions()
        assert versions == []
    finally:
        await client.close()


async def test_spawn_sends_correct_fields(aiohttp_server: Any) -> None:
    """spawn sends correct JSON body."""
    received: list[dict] = []

    app = web.Application()

    async def handle_spawn(request: web.Request) -> web.Response:
        body = await request.json()
        received.append(body)
        return web.json_response({"ok": True, "container-id": "c1", "generation": 1})

    app.router.add_post("/spawn", handle_spawn)
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        await client.spawn(
            spec_hash="sha256:" + "c" * 64,
            generation=2,
            artifact_path="gen-2",
        )
        assert len(received) == 1
        assert received[0]["spec-hash"] == "sha256:" + "c" * 64
        assert received[0]["generation"] == 2
        assert received[0]["artifact-path"] == "gen-2"
    finally:
        await client.close()


async def test_get_versions_empty_history(aiohttp_server: Any) -> None:
    """get_versions returns empty list for empty history."""
    app = await make_supervisor_app(versions_response=[])
    server = await aiohttp_server(app)
    client = SupervisorClient(base_url=str(server.make_url("/")))
    try:
        versions = await client.get_versions()
        assert versions == []
    finally:
        await client.close()


from typing import Any