#!/usr/bin/env python3
"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger().bind(component="prime")

BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Cambrian Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        url = f"{self.base_url}{path}"
        for i, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as exc:
                if i < len(BACKOFF_DELAYS) - 1:
                    logger.warning(
                        "supervisor GET failed, retrying",
                        url=url,
                        error=str(exc),
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request with exponential backoff."""
        url = f"{self.base_url}{path}"
        for i, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
            except aiohttp.ClientError as exc:
                if i < len(BACKOFF_DELAYS) - 1:
                    logger.warning(
                        "supervisor POST failed, retrying",
                        url=url,
                        error=str(exc),
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of GenerationRecord dicts."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception as exc:
            logger.error("get_versions failed", error=str(exc))
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
        """POST /spawn — start a Test Rig container."""
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