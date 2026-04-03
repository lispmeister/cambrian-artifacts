"""Prime — entry point, HTTP server, and main generation loop orchestrator."""
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
    """GET /stats — status information."""
    global _prime_generation, _status, _start_time
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Run the generation loop as a background task."""
    global _status

    # Check for required environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("missing_api_key", component="prime", event_detail="ANTHROPIC_API_KEY not set")
        return

    try:
        from src.loop import generation_loop
        await generation_loop()
    except Exception as e:
        log.error("generation_loop_error", component="prime", error=str(e))
        _status = "idle"


async def main() -> None:
    """Main entry point — start HTTP server then begin generation loop."""
    global _prime_generation, _status

    generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    port = 8401

    log.info("prime_starting", component="prime", generation=generation, port=port)

    app = make_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=generation, port=port)

    # Start generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        # Keep running until loop completes or interrupted
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
