"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()


class SupervisorClient:
    """Client for the Cambrian Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        delay = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}{path}") as resp:
                        return await resp.json()
            except Exception as exc:
                log.warning(
                    "supervisor_get_error",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request with exponential backoff."""
        delay = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}{path}",
                        json=body,
                    ) as resp:
                        return await resp.json()
            except Exception as exc:
                log.warning(
                    "supervisor_post_error",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of GenerationRecords."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        result = await self._get("/stats")
        return result if isinstance(result, dict) else {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — start a Test Rig for the given artifact."""
        return await self._post("/spawn", {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        })

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a viable generation."""
        return await self._post("/promote", {"generation": generation})

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a failed generation."""
        return await self._post("/rollback", {"generation": generation})
