"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """HTTP client for the Supervisor API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """GET request with exponential backoff retry."""
        url = f"{self.base_url}{path}"
        delay_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                delay = BACKOFF_DELAYS[min(delay_idx, len(BACKOFF_DELAYS) - 1)]
                log.warning(
                    "supervisor_get_error",
                    component="prime",
                    url=url,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay_idx += 1

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST request with exponential backoff retry."""
        url = f"{self.base_url}{path}"
        delay_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body) as resp:
                        resp.raise_for_status()
                        return await resp.json()  # type: ignore[no-any-return]
            except Exception as exc:
                delay = BACKOFF_DELAYS[min(delay_idx, len(BACKOFF_DELAYS) - 1)]
                log.warning(
                    "supervisor_post_error",
                    component="prime",
                    url=url,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay_idx += 1

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — return list of generation records."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — return supervisor stats."""
        result = await self._get("/stats")
        return result  # type: ignore[no-any-return]

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request Test Rig creation."""
        body = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        return await self._post("/spawn", body)

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        return await self._post("/promote", {"generation": generation})

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        return await self._post("/rollback", {"generation": generation})
