#!/usr/bin/env python3
"""Supervisor API client."""

import asyncio
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger().bind(component="prime")

BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as e:
                wait = BACKOFF_SEQUENCE[min(backoff_idx, len(BACKOFF_SEQUENCE) - 1)]
                logger.warning("Supervisor GET failed, retrying", url=url, error=str(e), wait=wait)
                await asyncio.sleep(wait)
                backoff_idx += 1

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        backoff_idx = 0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except Exception as e:
                wait = BACKOFF_SEQUENCE[min(backoff_idx, len(BACKOFF_SEQUENCE) - 1)]
                logger.warning("Supervisor POST failed, retrying", url=url, error=str(e), wait=wait)
                await asyncio.sleep(wait)
                backoff_idx += 1

    async def get_versions(self) -> list[dict[str, Any]]:
        result = await self._get("/versions")
        if isinstance(result, list):
            return result
        return []

    async def get_stats(self) -> dict[str, Any]:
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
        result = await self._post("/promote", {"generation": generation})
        if isinstance(result, dict):
            return result
        return {}

    async def rollback(self, generation: int) -> dict[str, Any]:
        result = await self._post("/rollback", {"generation": generation})
        if isinstance(result, dict):
            return result
        return {}