"""Prime — entry point, HTTP server, and main generation loop orchestrator."""
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
    """Create and return the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """GET /stats — status and identity."""
    global _prime_generation, _prime_status
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
    """Main entry point: start HTTP server, then run generation loop."""
    global _prime_generation, _prime_status

    # Configure structlog
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

    # Check if ANTHROPIC_API_KEY is set before starting loop
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning(
            "no_api_key_skipping_loop",
            component="prime",
            generation=_prime_generation,
        )
        # Stay alive serving /health and /stats
        while True:
            await asyncio.sleep(3600)
    else:
        loop_task = asyncio.create_task(run_generation_loop())
        await loop_task


if __name__ == "__main__":
    asyncio.run(main())
