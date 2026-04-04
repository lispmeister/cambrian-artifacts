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
        max_delay = 60.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}{path}", timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        else:
                            text = await resp.text()
                            log.warning(
                                "supervisor_get_error",
                                component="prime",
                                path=path,
                                status=resp.status,
                                body=text[:200],
                            )
                            return None
            except Exception as exc:
                log.warning(
                    "supervisor_unreachable",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request with exponential backoff."""
        delay = 1.0
        max_delay = 60.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}{path}",
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        result: dict[str, Any] = await resp.json()
                        if result.get("ok") is False:
                            log.warning(
                                "supervisor_post_nok",
                                component="prime",
                                path=path,
                                error=result.get("error", "unknown"),
                            )
                        return result
            except Exception as exc:
                log.warning(
                    "supervisor_unreachable",
                    component="prime",
                    path=path,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of GenerationRecords."""
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
        """POST /spawn — request Test Rig container."""
        return await self._post(
            "/spawn",
            {
                "spec-hash": spec_hash,
                "generation": generation,
                "artifact-path": artifact_path,
            },
        )

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        return await self._post("/promote", {"generation": generation})

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        return await self._post("/rollback", {"generation": generation})
