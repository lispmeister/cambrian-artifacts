"""Tests for Supervisor HTTP client (mock HTTP)."""
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestServer

from src import supervisor


@pytest.fixture(autouse=True)
async def reset_session():
    """Ensure each test starts with a fresh session."""
    await supervisor.close()
    yield
    await supervisor.close()


@pytest.fixture
async def mock_supervisor(aiohttp_server):
    """Create a mock Supervisor server."""
    versions_data = []
    spawn_response = {"ok": True, "container-id": "test", "generation": 1}

    async def handle_versions(request):
        return web.json_response(versions_data)

    async def handle_spawn(request):
        return web.json_response(spawn_response)

    async def handle_promote(request):
        body = await request.json()
        gen = body["generation"]
        return web.json_response({"ok": True, "generation": gen, "tag": f"gen-{gen}"})

    async def handle_rollback(request):
        body = await request.json()
        gen = body["generation"]
        return web.json_response({"ok": True, "generation": gen, "tag": f"gen-{gen}-failed"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    app.router.add_post("/spawn", handle_spawn)
    app.router.add_post("/promote", handle_promote)
    app.router.add_post("/rollback", handle_rollback)

    server = await aiohttp_server(app)

    # Point supervisor client at mock server
    original_url = supervisor.SUPERVISOR_URL
    supervisor.SUPERVISOR_URL = f"http://localhost:{server.port}"

    # Store mutable state on the fixture for test manipulation
    server.versions_data = versions_data
    server.spawn_response = spawn_response

    yield server

    supervisor.SUPERVISOR_URL = original_url


class TestGetVersions:
    async def test_returns_empty_list_when_no_records(self, mock_supervisor):
        result = await supervisor.get_versions()
        assert result == []

    async def test_returns_records(self, mock_supervisor):
        mock_supervisor.versions_data.extend([
            {"generation": 1, "outcome": "promoted"},
            {"generation": 2, "outcome": "failed"},
        ])
        result = await supervisor.get_versions()
        assert len(result) == 2
        assert result[0]["generation"] == 1


class TestSpawn:
    async def test_returns_ok(self, mock_supervisor):
        result = await supervisor.spawn(generation=1, artifact_path="gen-1", spec_hash="sha256:abc")
        assert result["ok"] is True
        assert result["generation"] == 1


class TestPromote:
    async def test_returns_ok_with_tag(self, mock_supervisor):
        result = await supervisor.promote(generation=1)
        assert result["ok"] is True
        assert result["tag"] == "gen-1"


class TestRollback:
    async def test_returns_ok_with_tag(self, mock_supervisor):
        result = await supervisor.rollback(generation=2)
        assert result["ok"] is True
        assert result["tag"] == "gen-2-failed"


class TestPollUntilTerminal:
    async def test_returns_when_outcome_not_in_progress(self, mock_supervisor):
        mock_supervisor.versions_data.extend([
            {"generation": 1, "outcome": "tested", "viability": {"status": "viable"}},
        ])
        result = await supervisor.poll_until_terminal(generation=1, interval=0.1)
        assert result["outcome"] == "tested"

    async def test_waits_while_in_progress(self, mock_supervisor):
        mock_supervisor.versions_data.extend([
            {"generation": 1, "outcome": "in_progress"},
        ])

        async def promote_after_delay():
            await asyncio.sleep(0.3)
            mock_supervisor.versions_data.clear()
            mock_supervisor.versions_data.append(
                {"generation": 1, "outcome": "tested", "viability": {"status": "viable"}}
            )

        asyncio.create_task(promote_after_delay())
        result = await supervisor.poll_until_terminal(generation=1, interval=0.1)
        assert result["outcome"] == "tested"
