"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

_BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        url = f"{self.base_url}{path}"
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        return await resp.json()
            except Exception as e:
                delay = _BACKOFF_SEQUENCE[min(backoff_idx, len(_BACKOFF_SEQUENCE) - 1)]
                log.warning("supervisor_get_error", component="prime", url=url, error=str(e), retry_in=delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Make a POST request with exponential backoff."""
        url = f"{self.base_url}{path}"
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        return await resp.json()
            except Exception as e:
                delay = _BACKOFF_SEQUENCE[min(backoff_idx, len(_BACKOFF_SEQUENCE) - 1)]
                log.warning("supervisor_post_error", component="prime", url=url, error=str(e), retry_in=delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        result = await self._get("/versions")
        if isinstance(result, list):
            return result
        return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        result = await self._get("/stats")
        if isinstance(result, dict):
            return result
        return {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — start a Test Rig container."""
        body = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        result = await self._post("/spawn", body)
        if isinstance(result, dict):
            return result
        return {}

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        result = await self._post("/promote", {"generation": generation})
        if isinstance(result, dict):
            return result
        return {}

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        result = await self._post("/rollback", {"generation": generation})
        if isinstance(result, dict):
            return result
        return {}
