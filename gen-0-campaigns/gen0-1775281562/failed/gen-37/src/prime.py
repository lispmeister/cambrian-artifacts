"""Prime — the organism. Entry point, HTTP server, main loop."""
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
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

# Prime's own generation number (fixed at startup from environment)
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_start_time: float = time.monotonic()
_status: str = "idle"


def make_app() -> web.Application:
    """Create the aiohttp application with health and stats endpoints."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — status endpoint."""
    uptime = int(time.monotonic() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Background task: run the generation loop."""
    global _status
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
    """Main entry point."""
    global _prime_generation

    # Validate required environment
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.error(
            "missing_api_key",
            component="prime",
            generation=_prime_generation,
            port=8401,
        )
        # Don't fatal-exit — still serve /health so Test Rig can check us

    log.info(
        "prime_starting",
        component="prime",
        generation=_prime_generation,
        port=8401,
    )

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()

    log.info(
        "prime_listening",
        component="prime",
        generation=_prime_generation,
        port=8401,
    )

    # Start generation loop as background task only if API key is present
    if api_key:
        loop_task = asyncio.create_task(run_generation_loop())
        try:
            await loop_task
        except Exception as exc:
            log.error(
                "generation_loop_failed",
                component="prime",
                generation=_prime_generation,
                port=8401,
                error=str(exc),
            )
    else:
        # No API key — just serve HTTP indefinitely
        await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
