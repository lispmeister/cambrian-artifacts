"""Supervisor API client for Prime."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

from src.models import GenerationRecord, SpawnResponse

logger = structlog.get_logger().bind(component="prime")

_BACKOFF_SCHEDULE = [1, 2, 4, 8, 16, 60]


class SupervisorError(Exception):
    """Raised when the Supervisor API returns an error."""
    pass


class SupervisorClient:
    """HTTP client for the Supervisor API."""

    def __init__(self, base_url: str = "http://host.docker.internal:8400") -> None:
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request_with_backoff(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make an HTTP request with exponential backoff on failure."""
        url = f"{self.base_url}{path}"
        log = logger.bind(method=method, url=url)
        attempt = 0

        while True:
            try:
                session = self._get_session()
                async with session.request(method, url, **kwargs) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    retry_after = int(e.headers.get("retry-after", "5")) if e.headers else 5
                    log.warning("Rate limited, waiting", retry_after=retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                log.error("HTTP error", status=e.status, error=str(e))
                raise SupervisorError(f"HTTP {e.status}: {e.message}") from e
            except (aiohttp.ClientError, OSError) as e:
                wait = _BACKOFF_SCHEDULE[min(attempt, len(_BACKOFF_SCHEDULE) - 1)]
                log.warning("Supervisor unreachable, retrying", attempt=attempt, wait=wait, error=str(e))
                await asyncio.sleep(wait)
                attempt += 1

    async def get_versions(self) -> list[GenerationRecord]:
        """GET /versions — retrieve all generation records."""
        data = await self._request_with_backoff("GET", "/versions")
        if not isinstance(data, list):
            return []
        records = []
        for item in data:
            try:
                records.append(GenerationRecord.model_validate(item))
            except Exception as e:
                logger.warning("Invalid generation record", error=str(e), item=item)
        return records

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — retrieve supervisor stats."""
        data = await self._request_with_backoff("GET", "/stats")
        return dict(data)

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request a new Test Rig container."""
        payload = {
            "spec-hash": spec_hash,
            "generation": generation,
            "artifact-path": artifact_path,
        }
        data = await self._request_with_backoff("POST", "/spawn", json=payload)
        return dict(data)

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a successful generation."""
        payload = {"generation": generation}
        data = await self._request_with_backoff("POST", "/promote", json=payload)
        return dict(data)

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a failed generation."""
        payload = {"generation": generation}
        data = await self._request_with_backoff("POST", "/rollback", json=payload)
        return dict(data)