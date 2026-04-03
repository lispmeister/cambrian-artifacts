"""Supervisor API client."""
from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger()


class SupervisorClient:
    """Async client for the Supervisor HTTP API."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request_with_backoff(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Make an HTTP request with exponential backoff on failure."""
        delays = [1, 2, 4, 8, 16, 60]
        attempt = 0
        while True:
            try:
                session = await self._get_session()
                url = f"{self._base_url}{path}"
                if method == "GET":
                    async with session.get(url) as resp:
                        return await resp.json()
                elif method == "POST":
                    async with session.post(url, json=json_data) as resp:
                        return await resp.json()
            except Exception as exc:
                delay = delays[min(attempt, len(delays) - 1)]
                logger.error(
                    "supervisor_request_failed",
                    component="prime",
                    path=path,
                    attempt=attempt,
                    delay=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions - retrieve generation history."""
        try:
            result = await self._request_with_backoff("GET", "/versions")
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats - retrieve supervisor stats."""
        result = await self._request_with_backoff("GET", "/stats")
        if isinstance(result, dict):
            return result
        return {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn - request Test Rig verification."""
        body = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        result = await self._request_with_backoff("POST", "/spawn", json_data=body)
        if isinstance(result, dict):
            return result
        return {"ok": False, "error": "unexpected response"}

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote - promote a viable generation."""
        body = {"generation": generation}
        result = await self._request_with_backoff("POST", "/promote", json_data=body)
        if isinstance(result, dict):
            return result
        return {"ok": False, "error": "unexpected response"}

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback - rollback a non-viable generation."""
        body = {"generation": generation}
        result = await self._request_with_backoff("POST", "/rollback", json_data=body)
        if isinstance(result, dict):
            return result
        return {"ok": False, "error": "unexpected response"}
