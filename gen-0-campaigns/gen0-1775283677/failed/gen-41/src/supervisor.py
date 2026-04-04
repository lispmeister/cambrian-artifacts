"""Supervisor API client."""

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        url = f"{self.base_url}{path}"
        delays = BACKOFF_DELAYS[:]
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                if not delays:
                    raise
                delay = delays.pop(0)
                log.warning(
                    "supervisor_get_retry",
                    component="prime",
                    url=url,
                    error=str(exc),
                    retry_after=delay,
                )
                await asyncio.sleep(delay)

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Make a POST request with exponential backoff."""
        url = f"{self.base_url}{path}"
        delays = BACKOFF_DELAYS[:]
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                if not delays:
                    raise
                delay = delays.pop(0)
                log.warning(
                    "supervisor_post_retry",
                    component="prime",
                    url=url,
                    error=str(exc),
                    retry_after=delay,
                )
                await asyncio.sleep(delay)

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception as exc:
            log.error(
                "get_versions_failed",
                component="prime",
                error=str(exc),
            )
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        return await self._get("/stats")  # type: ignore[return-value]

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request Test Rig container."""
        return await self._post(  # type: ignore[return-value]
            "/spawn",
            {
                "spec-hash": spec_hash,
                "generation": generation,
                "artifact-path": artifact_path,
            },
        )

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        return await self._post("/promote", {"generation": generation})  # type: ignore[return-value]

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        return await self._post("/rollback", {"generation": generation})  # type: ignore[return-value]
