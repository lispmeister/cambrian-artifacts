"""Prime — entry point, HTTP server, and main generation loop orchestrator."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

log = structlog.get_logger()

_start_time: float = time.monotonic()
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
    """GET /stats — status information."""
    uptime = int(time.monotonic() - _start_time)
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
            error=str(exc),
        )


async def main() -> None:
    """Main entry point: start HTTP server then run generation loop."""
    global _prime_status

    port = 8401
    generation = _prime_generation

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

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

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        log.error(
            "missing_api_key",
            component="prime",
            generation=generation,
            error="ANTHROPIC_API_KEY is not set",
        )
        # Keep serving health checks even without API key
        await asyncio.Event().wait()
        return

    loop_task = asyncio.create_task(run_generation_loop())
    try:
        await loop_task
    except Exception as exc:
        log.error(
            "generation_loop_failed",
            component="prime",
            generation=generation,
            error=str(exc),
        )

    # Keep serving after loop completes
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
