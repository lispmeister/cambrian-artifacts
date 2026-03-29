#!/usr/bin/env python3
"""Prime — the organism. Async HTTP server and generation loop."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

logger = structlog.get_logger().bind(component="prime")

_start_time = time.time()
_status = "idle"


def get_generation() -> int:
    """Get this Prime's own generation number from environment."""
    try:
        return int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    except (ValueError, TypeError):
        return 0


def make_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    """GET /stats — status information."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": get_generation(),
        "status": _status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Background task: run the generation loop."""
    global _status

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — generation loop disabled")
        return

    from src.generate import GenerationConfig, run_loop
    config = GenerationConfig.from_env()

    _status = "generating"
    try:
        await run_loop(config, status_callback=_set_status)
    except Exception as exc:
        logger.error("generation loop crashed", error=str(exc))
    finally:
        _status = "idle"


def _set_status(s: str) -> None:
    global _status
    _status = s


async def main() -> None:
    """Entry point: start HTTP server then begin generation loop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — running in health-only mode")

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()

    logger.info(
        "Prime HTTP server started",
        port=8401,
        generation=get_generation(),
    )

    # Start generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        # Run forever
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
    asyncio.run(main())