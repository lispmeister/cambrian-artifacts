"""Prime — the organism. Entry point, HTTP server, and main loop."""
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
    """Create and configure the aiohttp application."""
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
        log.error("generation_loop_error", error=str(exc), component="prime")


async def main() -> None:
    """Start HTTP server and generation loop."""
    global _prime_generation

    _prime_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    port = 8401

    # Validate required environment
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("anthropic_api_key_missing", component="prime",
                    event_detail="ANTHROPIC_API_KEY not set — generation loop will not start")

    log.info("prime_starting", component="prime", generation=_prime_generation, port=port)

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=_prime_generation, port=port)

    # Start generation loop as background task
    loop_task: asyncio.Task[None] | None = None
    if api_key:
        loop_task = asyncio.create_task(run_generation_loop())

    # Keep server running
    try:
        if loop_task is not None:
            await loop_task
        else:
            # No API key — just serve HTTP indefinitely
            while True:
                await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
    asyncio.run(main())
