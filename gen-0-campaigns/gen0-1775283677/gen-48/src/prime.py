"""Prime — the organism. Entry point and HTTP server."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

log = structlog.get_logger()

_start_time: float = time.time()
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_prime_status: str = "idle"


def make_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — status information."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _prime_status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Run the generation loop as a background task."""
    global _prime_status
    try:
        from src.loop import generation_loop
        await generation_loop()
    except Exception as exc:
        log.error(
            "generation_loop_error",
            component="prime",
            error=str(exc),
        )


async def main() -> None:
    """Main entry point."""
    global _prime_generation, _prime_status

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    port = 8401
    _prime_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))

    log.info(
        "prime_starting",
        component="prime",
        generation=_prime_generation,
        port=port,
    )

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.warning(
            "anthropic_api_key_missing",
            component="prime",
            generation=_prime_generation,
        )

    # Start HTTP server first
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

    # Run generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
