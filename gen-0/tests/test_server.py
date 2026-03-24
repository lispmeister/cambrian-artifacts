"""Phase 0 validation: gen-0 server responds correctly on /health."""
import pytest
import aiohttp
import asyncio


@pytest.mark.asyncio
async def test_health_returns_200():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8401/health") as resp:
            assert resp.status == 200


@pytest.mark.asyncio
async def test_health_returns_ok():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8401/health") as resp:
            body = await resp.json()
            assert body.get("ok") is True
