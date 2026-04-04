"""Prime — the organism. Entry point, HTTP server, main loop."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

log = structlog.get_logger()

_start_time = time.time()
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_prime_status: str = "idle"


def make_app() -> web.Application:
    """Create and return the aiohttp Application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    """GET /stats — generation stats."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _prime_status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Background task: run the generation loop."""
    global _prime_status
    try:
        from src.loop import generation_loop
        await generation_loop()
    except Exception as exc:
        log.error(
            "generation_loop_error",
            component="prime",
            generation=_prime_generation,
            port=8401,
            error=str(exc),
        )


async def main() -> None:
    """Start the HTTP server and generation loop."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    port = 8401

    log.info(
        "prime_starting",
        component="prime",
        generation=_prime_generation,
        port=port,
    )

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info(
        "prime_listening",
        component="prime",
        generation=_prime_generation,
        port=port,
    )

    loop_task = asyncio.create_task(run_generation_loop())

    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
