"""Supervisor API client."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(component="prime")

BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """GET request with exponential backoff on network errors."""
        url = f"{self.base_url}{path}"
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as e:
                log.warning(
                    "Supervisor GET failed, retrying",
                    url=url,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
            except Exception as e:
                log.error("Unexpected error on GET", url=url, error=str(e))
                raise

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """POST request with exponential backoff on network errors."""
        url = f"{self.base_url}{path}"
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as e:
                log.warning(
                    "Supervisor POST failed, retrying",
                    url=url,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
            except Exception as e:
                log.error("Unexpected error on POST", url=url, error=str(e))
                raise

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of GenerationRecord dicts."""
        result = await self._get("/versions")
        return result if isinstance(result, list) else []

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
        """POST /spawn."""
        body: dict[str, Any] = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        result = await self._post("/spawn", body)
        return result if isinstance(result, dict) else {"ok": False}

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote."""
        result = await self._post("/promote", {"generation": generation})
        return result if isinstance(result, dict) else {"ok": False}

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback."""
        result = await self._post("/rollback", {"generation": generation})
        return result if isinstance(result, dict) else {"ok": False}