#!/usr/bin/env python3
"""Prime — the organism. An async HTTP server and code generation loop."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger(component="prime")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

START_TIME: float = time.time()
GENERATION: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
STATUS: str = "idle"


def get_uptime() -> int:
    return int(time.time() - START_TIME)


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "generation": GENERATION,
            "status": STATUS,
            "uptime": get_uptime(),
        }
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


async def run_generation_loop() -> None:
    """Background task: run the generation loop."""
    global STATUS

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not set — generation loop will not run")
        return

    from src.generate import GenerationLoop

    loop = GenerationLoop()
    await loop.run()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    global STATUS

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — generation loop disabled")

    log.info("Starting Prime", generation=GENERATION)

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()

    log.info("HTTP server started", port=8401)

    # Start generation loop as background task
    if api_key:
        asyncio.create_task(run_generation_loop())
    else:
        log.warning("Skipping generation loop — no API key")

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())