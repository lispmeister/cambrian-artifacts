"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger()


class SupervisorClient:
    """Client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _get(self, path: str) -> Any:
        """Make a GET request with exponential backoff."""
        delays = [1, 2, 4, 8, 16, 60]
        last_exc: Exception | None = None
        for delay in delays:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.base_url}{path}", timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        return await resp.json()
            except Exception as exc:
                last_exc = exc
                log.warning("supervisor_get_failed", component="prime", path=path, error=str(exc))
                await asyncio.sleep(delay)
        raise RuntimeError(f"Supervisor unreachable at {path}: {last_exc}")

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        """Make a POST request with exponential backoff."""
        delays = [1, 2, 4, 8, 16, 60]
        last_exc: Exception | None = None
        for delay in delays:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}{path}",
                        json=body,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        return await resp.json()
            except Exception as exc:
                last_exc = exc
                log.warning("supervisor_post_failed", component="prime", path=path, error=str(exc))
                await asyncio.sleep(delay)
        raise RuntimeError(f"Supervisor unreachable at {path}: {last_exc}")

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — get all generation records."""
        try:
            result = await self._get("/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — get supervisor stats."""
        result = await self._get("/stats")
        return result  # type: ignore[return-value]

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request test rig for artifact."""
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
