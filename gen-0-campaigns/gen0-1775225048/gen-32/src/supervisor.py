"""Supervisor API client."""
from __future__ import annotations

from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/versions") as resp:
                if resp.status != 200:
                    log.warning("get_versions_non_200", component="prime", status=resp.status)
                    return []
                data = await resp.json()
                if not isinstance(data, list):
                    return []
                return data

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns Supervisor stats."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/stats") as resp:
                resp.raise_for_status()
                return await resp.json()

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request Test Rig spawn."""
        payload = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/spawn", json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/promote",
                json={"generation": generation}
            ) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/rollback",
                json={"generation": generation}
            ) as resp:
                resp.raise_for_status()
                return await resp.json()
