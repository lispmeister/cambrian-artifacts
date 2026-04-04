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
    ]
)

log = structlog.get_logger()

# Global state
_start_time: float = time.time()
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_status: str = "idle"


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
    """GET /stats — generation stats."""
    uptime = int(time.time() - _start_time)
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
    except Exception as e:
        log.error("generation_loop_error", error=str(e), component="prime")
    finally:
        _status = "idle"


async def main() -> None:
    """Main entry point: start HTTP server then run generation loop."""
    generation = _prime_generation
    port = 8401

    log.info("prime_starting", component="prime", generation=generation, port=port)

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=generation, port=port)

    # Run generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    # Validate required env vars
    if not os.environ.get("ANTHROPIC_API_KEY"):
        import sys
        log.error("missing_api_key", component="prime", error="ANTHROPIC_API_KEY is required")
        sys.exit(1)

    asyncio.run(main())
