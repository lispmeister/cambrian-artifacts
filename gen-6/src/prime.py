#!/usr/bin/env python3
"""Prime — the organism. Entry point, HTTP server, and main generation loop."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

logger = structlog.get_logger().bind(component="prime")

_start_time = time.monotonic()
_status = "idle"
_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))


def make_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    """Liveness check — always returns 200."""
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    """Stats endpoint."""
    uptime = int(time.monotonic() - _start_time)
    return web.json_response({
        "generation": _generation,
        "status": _status,
        "uptime": uptime,
    })


def set_status(new_status: str) -> None:
    """Update global status."""
    global _status
    _status = new_status


async def generation_loop(app: web.Application) -> None:
    """Main generation loop — runs as a background task."""
    from src.loop import run_generation_loop
    await run_generation_loop(set_status)


async def on_startup(app: web.Application) -> None:
    """Start the generation loop as a background task on app startup."""
    if os.environ.get("CAMBRIAN_NO_LOOP", "").lower() in ("1", "true", "yes"):
        logger.info("generation loop disabled", reason="CAMBRIAN_NO_LOOP set")
        return
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — generation loop disabled")
        return
    task = asyncio.create_task(generation_loop(app))
    app["generation_task"] = task


async def on_cleanup(app: web.Application) -> None:
    """Cancel generation loop on shutdown."""
    task = app.get("generation_task")
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def create_app() -> web.Application:
    """Create app with lifecycle hooks."""
    app = make_app()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    """Entry point."""
    import sys

    # Validate required env vars
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — generation loop will be disabled")

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )

    app = create_app()
    port = int(os.environ.get("PRIME_PORT", "8401"))
    logger.info("starting prime", port=port, generation=_generation)
    web.run_app(app, host="0.0.0.0", port=port, access_log=None)


if __name__ == "__main__":
    main()