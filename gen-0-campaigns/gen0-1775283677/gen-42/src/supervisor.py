"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()


class SupervisorClient:
    """HTTP client for the Supervisor API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        url = f"{self.base_url}/versions"
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, list):
                                return data
                            return []
                        else:
                            log.warning("supervisor_versions_error", component="prime",
                                        status=resp.status)
                            return []
            except aiohttp.ClientError as e:
                log.warning("supervisor_unreachable", component="prime",
                            url=url, error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        url = f"{self.base_url}/stats"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {}
        except aiohttp.ClientError as e:
            log.warning("supervisor_stats_error", component="prime", error=str(e))
            return {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request verification of a generated artifact."""
        url = f"{self.base_url}/spawn"
        payload = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                        return data
            except aiohttp.ClientError as e:
                log.warning("supervisor_spawn_error", component="prime",
                            error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a viable generation."""
        url = f"{self.base_url}/promote"
        payload = {"generation": generation}
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                        return data
            except aiohttp.ClientError as e:
                log.warning("supervisor_promote_error", component="prime",
                            error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a non-viable generation."""
        url = f"{self.base_url}/rollback"
        payload = {"generation": generation}
        backoff = 1.0
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as resp:
                        data = await resp.json()
                        return data
            except aiohttp.ClientError as e:
                log.warning("supervisor_rollback_error", component="prime",
                            error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
