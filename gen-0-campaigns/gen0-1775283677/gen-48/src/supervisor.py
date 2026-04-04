"""Supervisor API client."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()

_BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with backoff retry."""
        url = f"{self.base_url}{path}"
        delay_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                delay = _BACKOFF_DELAYS[min(delay_idx, len(_BACKOFF_DELAYS) - 1)]
                log.warning(
                    "supervisor_get_error",
                    component="prime",
                    url=url,
                    error=str(exc),
                    retry_in=delay,
                )
                await asyncio.sleep(delay)
                delay_idx += 1

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Make a POST request with backoff retry."""
        url = f"{self.base_url}{path}"
        delay_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as exc:
                delay = _BACKOFF_DELAYS[min(delay_idx, len(_BACKOFF_DELAYS) - 1)]
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
        return result  # type: ignore[return-value]

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request a new Test Rig container."""
        body = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        result = await self._post("/spawn", body)
        return result  # type: ignore[return-value]

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        result = await self._post("/promote", {"generation": generation})
        return result  # type: ignore[return-value]

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        result = await self._post("/rollback", {"generation": generation})
        return result  # type: ignore[return-value]
