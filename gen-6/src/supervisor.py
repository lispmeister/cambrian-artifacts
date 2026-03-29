#!/usr/bin/env python3
"""Supervisor API client."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger().bind(component="prime")

BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 60]


class SupervisorClient:
    """HTTP client for the Supervisor API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def get_versions(self) -> list[dict[str, Any]]:
        """GET /versions — returns list of generation records."""
        result = await self._get_with_backoff("/versions", default=[])
        if isinstance(result, list):
            return result
        return []

    async def get_stats(self) -> dict[str, Any]:
        """GET /stats — returns supervisor stats."""
        result = await self._get_with_backoff("/stats", default={})
        if isinstance(result, dict):
            return result
        return {}

    async def spawn(
        self,
        spec_hash: str,
        generation: int,
        artifact_path: str,
    ) -> dict[str, Any]:
        """POST /spawn — request test rig spawn."""
        return await self._post_with_backoff(
            "/spawn",
            {
                "spec-hash": spec_hash,
                "generation": generation,
                "artifact-path": artifact_path,
            },
        )

    async def promote(self, generation: int) -> dict[str, Any]:
        """POST /promote — promote a generation."""
        return await self._post_with_backoff("/promote", {"generation": generation})

    async def rollback(self, generation: int) -> dict[str, Any]:
        """POST /rollback — rollback a generation."""
        return await self._post_with_backoff("/rollback", {"generation": generation})

    async def _get_with_backoff(
        self, path: str, default: Any = None
    ) -> Any:
        """GET request with exponential backoff on network errors."""
        url = f"{self.base_url}{path}"
        for attempt, delay in enumerate(BACKOFF_SEQUENCE):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        else:
                            logger.warning(
                                "supervisor GET error",
                                path=path,
                                status=resp.status,
                            )
                            return default
            except aiohttp.ClientError as e:
                logger.warning(
                    "supervisor GET network error",
                    path=path,
                    error=str(e),
                    attempt=attempt,
                    next_delay=delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error("supervisor GET unexpected error", path=path, error=str(e))
                raise

        raise RuntimeError(f"Supervisor unreachable after retries: GET {path}")

    async def _post_with_backoff(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST request with exponential backoff on network errors."""
        url = f"{self.base_url}{path}"
        for attempt, delay in enumerate(BACKOFF_SEQUENCE):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=body) as resp:
                        data: dict[str, Any] = await resp.json()
                        if not data.get("ok", False):
                            logger.warning(
                                "supervisor POST returned not-ok",
                                path=path,
                                response=data,
                            )
                        return data
            except aiohttp.ClientError as e:
                logger.warning(
                    "supervisor POST network error",
                    path=path,
                    error=str(e),
                    attempt=attempt,
                    next_delay=delay,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.error("supervisor POST unexpected error", path=path, error=str(e))
                raise

        raise RuntimeError(f"Supervisor unreachable after retries: POST {path}")