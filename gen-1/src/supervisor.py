"""Supervisor HTTP API client with exponential backoff."""
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp
import structlog

log = structlog.get_logger(component="prime")

SUPERVISOR_URL = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://localhost:8400")


class SupervisorError(Exception):
    pass


async def _request(
    method: str, path: str, json_body: dict[str, Any] | None = None
) -> dict[str, Any] | list[Any]:
    """Make an HTTP request to the Supervisor with exponential backoff."""
    url = f"{SUPERVISOR_URL}{path}"
    backoff = 1.0
    max_backoff = 60.0

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                if method == "GET":
                    async with session.get(url) as resp:
                        return await resp.json()
                else:
                    async with session.post(url, json=json_body) as resp:
                        return await resp.json()
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


async def poll_until_terminal(generation: int, interval: float = 2.0) -> dict[str, Any]:
    """Poll GET /versions until the generation record has a terminal outcome."""
    terminal = {"promoted", "failed", "timeout"}
    while True:
        records = await get_versions()
        for record in records:
            if record.get("generation") == generation and record.get("outcome") in terminal:
                return record
        await asyncio.sleep(interval)
