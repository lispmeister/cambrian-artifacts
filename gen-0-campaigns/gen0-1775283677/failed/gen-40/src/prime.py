"""Prime — the organism. Entry point, HTTP server, and main loop."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

# Module-level state
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_prime_status: str = "idle"
_prime_start_time: float = time.time()


def make_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — generation, status, uptime."""
    uptime = int(time.time() - _prime_start_time)
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
            error=str(exc),
        )


async def main() -> None:
    """Main entry point: start HTTP server, then run generation loop."""
    global _prime_generation, _prime_status

    port = 8401
    generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    _prime_generation = generation

    log.info(
        "prime_starting",
        component="prime",
        generation=generation,
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
        generation=generation,
        port=port,
    )

    # Check if ANTHROPIC_API_KEY is available before starting loop
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning(
            "no_api_key",
            component="prime",
            generation=generation,
            event_detail="ANTHROPIC_API_KEY not set, generation loop disabled",
        )
        # Keep server running
        while True:
            await asyncio.sleep(3600)
    else:
        loop_task = asyncio.create_task(run_generation_loop())
        try:
            await loop_task
        except asyncio.CancelledError:
            pass

    await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
