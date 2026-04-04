"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """HTTP client for the Supervisor API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}{path}",
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        return await resp.json()
            except Exception as exc:
                wait = BACKOFF_SEQUENCE[min(backoff_idx, len(BACKOFF_SEQUENCE) - 1)]
                log.warning(
                    "supervisor_get_error",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=wait,
                )
                backoff_idx += 1
                await asyncio.sleep(wait)
                if backoff_idx > len(BACKOFF_SEQUENCE):
                    raise

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Make a POST request with exponential backoff."""
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}{path}",
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        return await resp.json()
            except Exception as exc:
                wait = BACKOFF_SEQUENCE[min(backoff_idx, len(BACKOFF_SEQUENCE) - 1)]
                log.warning(
                    "supervisor_post_error",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=wait,
                )
                backoff_idx += 1
                await asyncio.sleep(wait)
                if backoff_idx > len(BACKOFF_SEQUENCE):
                    raise

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        return await self._get("/stats")

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request Test Rig spawn."""
        return await self._post("/spawn", {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        })

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        return await self._post("/promote", {"generation": generation})

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        return await self._post("/rollback", {"generation": generation})
