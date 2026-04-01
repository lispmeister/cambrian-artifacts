#!/usr/bin/env python3
"""Supervisor API client."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger()

BACKOFF_DELAYS = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """Client for the Cambrian Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._log = logger.bind(component="prime", supervisor_url=self.base_url)

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        url = f"{self.base_url}/versions"
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, list):
                                return data
                            return []
                        self._log.warning("get_versions non-200", status=resp.status)
                        return []
            except Exception as e:
                self._log.warning("get_versions failed", attempt=attempt, error=str(e))
                await asyncio.sleep(delay)
        return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        url = f"{self.base_url}/stats"
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        self._log.warning("get_stats non-200", status=resp.status)
                        return {}
            except Exception as e:
                self._log.warning("get_stats failed", attempt=attempt, error=str(e))
                await asyncio.sleep(delay)
        return {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request Test Rig spawn."""
        url = f"{self.base_url}/spawn"
        payload = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        data = await resp.json()
                        if resp.status == 200 and data.get("ok"):
                            return data
                        self._log.warning("spawn failed", status=resp.status, data=data)
                        return data
            except Exception as e:
                self._log.warning("spawn error", attempt=attempt, error=str(e))
                await asyncio.sleep(delay)
        return {"ok": False, "error": "exhausted retries"}

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        url = f"{self.base_url}/promote"
        payload = {"generation": generation}
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        data = await resp.json()
                        if resp.status == 200 and data.get("ok"):
                            return data
                        self._log.warning("promote failed", status=resp.status, data=data)
                        return data
            except Exception as e:
                self._log.warning("promote error", attempt=attempt, error=str(e))
                await asyncio.sleep(delay)
        return {"ok": False, "error": "exhausted retries"}

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        url = f"{self.base_url}/rollback"
        payload = {"generation": generation}
        for attempt, delay in enumerate(BACKOFF_DELAYS + [60] * 100):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        data = await resp.json()
                        if resp.status == 200 and data.get("ok"):
                            return data
                        self._log.warning("rollback failed", status=resp.status, data=data)
                        return data
            except Exception as e:
                self._log.warning("rollback error", attempt=attempt, error=str(e))
                await asyncio.sleep(delay)
        return {"ok": False, "error": "exhausted retries"}
