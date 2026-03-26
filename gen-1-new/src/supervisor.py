"""Supervisor HTTP API client with exponential backoff."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(component="prime")

# Default: host.docker.internal because Prime runs inside a Docker container
SUPERVISOR_URL = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")

_session: aiohttp.ClientSession | None = None


class SupervisorError(Exception):
    pass


def _get_session() -> aiohttp.ClientSession:
    """Get or create a shared ClientSession."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


async def close() -> None:
    """Close the shared session. Call on shutdown."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None


async def _request(
    method: str, path: str, json_body: dict[str, Any] | None = None
) -> dict[str, Any] | list[Any]:
    """Make an HTTP request to the Supervisor with exponential backoff.

    Retries indefinitely on network errors — Prime must not proceed without
    the Supervisor (no self-promotion).
    """
    url = f"{SUPERVISOR_URL}{path}"
    backoff = 1.0
    max_backoff = 60.0

    while True:
        try:
            session = _get_session()
            if method == "GET":
                async with session.get(url) as resp:
                    return await resp.json()  # type: ignore[return-value]
            else:
                async with session.post(url, json=json_body) as resp:
                    return await resp.json()  # type: ignore[return-value]
        except (aiohttp.ClientError, OSError) as e:
            log.warning(
                "supervisor_unreachable",
                url=url,
                error=str(e),
                retry_in=backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


async def get_versions() -> list[dict[str, Any]]:
    """GET /versions — returns all generation records."""
    result = await _request("GET", "/versions")
    return result if isinstance(result, list) else []


async def spawn(generation: int, artifact_path: str, spec_hash: str) -> dict[str, Any]:
    """POST /spawn — start Test Rig for an artifact."""
    result = await _request("POST", "/spawn", {
        "generation": generation,
        "artifact-path": artifact_path,
        "spec-hash": spec_hash,
    })
    return result if isinstance(result, dict) else {}


async def promote(generation: int) -> dict[str, Any]:
    """POST /promote — promote a viable generation."""
    result = await _request("POST", "/promote", {"generation": generation})
    return result if isinstance(result, dict) else {}


async def rollback(generation: int) -> dict[str, Any]:
    """POST /rollback — roll back a failed generation."""
    result = await _request("POST", "/rollback", {"generation": generation})
    return result if isinstance(result, dict) else {}


async def poll_until_tested(generation: int, interval: float = 2.0) -> dict[str, Any]:
    """Poll GET /versions until the generation record's outcome is no longer in_progress.

    Returns the generation record once outcome transitions out of in_progress.
    """
    while True:
        records = await get_versions()
        for record in records:
            if record.get("generation") == generation and record.get("outcome") != "in_progress":
                return record
        await asyncio.sleep(interval)
