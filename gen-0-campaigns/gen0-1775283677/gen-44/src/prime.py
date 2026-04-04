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
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — status and uptime."""
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
    """Start HTTP server then run generation loop."""
    global _prime_generation
    _prime_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))

    # Check required env vars
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("anthropic_api_key_missing", component="prime",
                    message="ANTHROPIC_API_KEY not set; generation loop will be skipped")

    port = 8401
    log.info("prime_starting", component="prime", generation=_prime_generation, port=port)

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=_prime_generation, port=port)

    if api_key:
        loop_task = asyncio.create_task(run_generation_loop())
        try:
            await loop_task
        except Exception as exc:
            log.error("generation_loop_failed", error=str(exc), component="prime")
    else:
        # No API key — just serve HTTP forever
        await asyncio.Event().wait()


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
