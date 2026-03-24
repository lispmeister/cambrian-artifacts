"""Phase 0 validation: gen-0 server /health responds correctly (in-process test)."""
import pytest
from aiohttp.test_utils import TestClient, TestServer
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.server import make_app


@pytest.fixture
async def client(aiohttp_client):  # type: ignore[no-untyped-def]
    return await aiohttp_client(make_app())


async def test_health_returns_200(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.get("/health")
    assert resp.status == 200


async def test_health_returns_ok(client) -> None:  # type: ignore[no-untyped-def]
    resp = await client.get("/health")
    body = await resp.json()
    assert body.get("ok") is True
