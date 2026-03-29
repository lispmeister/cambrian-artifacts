#!/usr/bin/env python3
"""Prime — the organism. Async HTTP server and generation loop."""

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

logger = structlog.get_logger().bind(component="prime")

START_TIME = time.time()
_status = "idle"
_generation_number = int(os.environ.get("CAMBRIAN_GENERATION", "0"))


def get_status() -> str:
    return _status


def set_status(s: str) -> None:
    global _status
    _status = s


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    uptime = int(time.time() - START_TIME)
    return web.json_response({
        "generation": _generation_number,
        "status": get_status(),
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Background task: the generation loop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — generation loop will not run")
        return

    from src.generate import run_loop
    await run_loop(set_status)


async def main() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — running in health-only mode")

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()

    log = logger.bind(generation=_generation_number)
    log.info("Prime HTTP server started", port=8401)

    # Start generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        await asyncio.Event().wait()
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())