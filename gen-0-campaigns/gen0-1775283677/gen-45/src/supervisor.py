"""Supervisor API client."""
from __future__ import annotations

import json
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of GenerationRecord dicts."""
        url = f"{self.base_url}/versions"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    log.warning(
                        "supervisor_get_versions_error",
                        component="prime",
                        status=resp.status,
                    )
                    return []
                data = await resp.json()
                if isinstance(data, list):
                    return data
                return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns Supervisor stats."""
        url = f"{self.base_url}/stats"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()  # type: ignore[no-any-return]

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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()  # type: ignore[no-any-return]

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        url = f"{self.base_url}/promote"
        payload = {"generation": generation}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()  # type: ignore[no-any-return]

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        url = f"{self.base_url}/rollback"
        payload = {"generation": generation}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()  # type: ignore[no-any-return]
