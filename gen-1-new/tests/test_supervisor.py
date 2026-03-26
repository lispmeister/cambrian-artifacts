"""Tests for Supervisor HTTP client using a mock aiohttp server."""
from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from src import supervisor


@pytest.fixture(autouse=True)
async def reset_session():
    """Ensure each test starts with a fresh session and a clean module state."""
    await supervisor.close()
    yield
    await supervisor.close()


@pytest.fixture
async def mock_supervisor_server(aiohttp_server):
    """Spin up a minimal mock Supervisor server and point the supervisor module at it."""
    # Mutable state shared between handler closures and test code
    state: dict = {
        "versions": [],
        "spawn_response": {"ok": True, "container-id": "test-ctr", "generation": 1},
    }

    async def handle_versions(request: web.Request) -> web.Response:
        return web.json_response(state["versions"])

    async def handle_spawn(request: web.Request) -> web.Response:
        return web.json_response(state["spawn_response"])

    async def handle_promote(request: web.Request) -> web.Response:
        body = await request.json()
        gen = body["generation"]
        return web.json_response({"ok": True, "generation": gen, "tag": f"gen-{gen}"})

    async def handle_rollback(request: web.Request) -> web.Response:
        body = await request.json()
        gen = body["generation"]
        return web.json_response({"ok": True, "generation": gen, "tag": f"gen-{gen}-failed"})

    app = web.Application()
    app.router.add_get("/versions", handle_versions)
    app.router.add_post("/spawn", handle_spawn)
    app.router.add_post("/promote", handle_promote)
    app.router.add_post("/rollback", handle_rollback)

    server = await aiohttp_server(app)

    original_url = supervisor.SUPERVISOR_URL
    supervisor.SUPERVISOR_URL = f"http://localhost:{server.port}"

    # Expose state dict so tests can mutate it
    server.state = state
    yield server

    supervisor.SUPERVISOR_URL = original_url


class TestGetVersions:
    async def test_returns_empty_list_when_no_records(self, mock_supervisor_server):
        result = await supervisor.get_versions()
        assert result == []

    async def test_returns_list_of_records(self, mock_supervisor_server):
        mock_supervisor_server.state["versions"].extend([
            {"generation": 1, "outcome": "promoted"},
            {"generation": 2, "outcome": "failed"},
        ])
        result = await supervisor.get_versions()
        assert len(result) == 2
        assert result[0]["generation"] == 1
        assert result[1]["outcome"] == "failed"


class TestSpawn:
    async def test_returns_ok(self, mock_supervisor_server):
        result = await supervisor.spawn(generation=1, artifact_path="gen-1", spec_hash="sha256:abc")
        assert result["ok"] is True

    async def test_returns_generation(self, mock_supervisor_server):
        result = await supervisor.spawn(generation=1, artifact_path="gen-1", spec_hash="sha256:abc")
        assert result["generation"] == 1

    async def test_returns_container_id(self, mock_supervisor_server):
        result = await supervisor.spawn(generation=1, artifact_path="gen-1", spec_hash="sha256:abc")
        assert "container-id" in result


class TestPromote:
    async def test_returns_ok(self, mock_supervisor_server):
        result = await supervisor.promote(generation=1)
        assert result["ok"] is True

    async def test_returns_correct_generation(self, mock_supervisor_server):
        result = await supervisor.promote(generation=3)
        assert result["generation"] == 3

    async def test_returns_tag(self, mock_supervisor_server):
        result = await supervisor.promote(generation=1)
        assert result["tag"] == "gen-1"


class TestRollback:
    async def test_returns_ok(self, mock_supervisor_server):
        result = await supervisor.rollback(generation=2)
        assert result["ok"] is True

    async def test_returns_failed_tag(self, mock_supervisor_server):
        result = await supervisor.rollback(generation=2)
        assert result["tag"] == "gen-2-failed"

    async def test_returns_correct_generation(self, mock_supervisor_server):
        result = await supervisor.rollback(generation=5)
        assert result["generation"] == 5


class TestPollUntilTested:
    async def test_returns_immediately_when_not_in_progress(self, mock_supervisor_server):
        mock_supervisor_server.state["versions"].append(
            {"generation": 1, "outcome": "tested", "viability": {"status": "viable"}}
        )
        result = await supervisor.poll_until_tested(generation=1, interval=0.05)
        assert result["outcome"] == "tested"

    async def test_waits_while_in_progress(self, mock_supervisor_server):
        mock_supervisor_server.state["versions"].append(
            {"generation": 1, "outcome": "in_progress"}
        )

        async def transition_to_tested():
            await asyncio.sleep(0.2)
            mock_supervisor_server.state["versions"].clear()
            mock_supervisor_server.state["versions"].append(
                {"generation": 1, "outcome": "tested", "viability": {"status": "viable"}}
            )

        asyncio.create_task(transition_to_tested())
        result = await supervisor.poll_until_tested(generation=1, interval=0.05)
        assert result["outcome"] == "tested"

    async def test_returns_promoted_record(self, mock_supervisor_server):
        mock_supervisor_server.state["versions"].append(
            {"generation": 2, "outcome": "promoted", "viability": {"status": "viable"}}
        )
        result = await supervisor.poll_until_tested(generation=2, interval=0.05)
        assert result["outcome"] == "promoted"

    async def test_ignores_other_generations(self, mock_supervisor_server):
        """Should skip records for other generation numbers."""
        mock_supervisor_server.state["versions"].extend([
            {"generation": 1, "outcome": "promoted"},
            {"generation": 2, "outcome": "in_progress"},
        ])

        async def finish_gen2():
            await asyncio.sleep(0.2)
            mock_supervisor_server.state["versions"].clear()
            mock_supervisor_server.state["versions"].extend([
                {"generation": 1, "outcome": "promoted"},
                {"generation": 2, "outcome": "tested", "viability": {"status": "viable"}},
            ])

        asyncio.create_task(finish_gen2())
        result = await supervisor.poll_until_tested(generation=2, interval=0.05)
        assert result["generation"] == 2
        assert result["outcome"] == "tested"
