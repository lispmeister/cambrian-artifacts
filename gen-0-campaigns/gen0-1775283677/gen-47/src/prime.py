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
    ]
)

log = structlog.get_logger()

# Global state
_start_time: float = time.time()
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
    global _prime_generation, _prime_status, _start_time
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _prime_status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Run the generation loop as a background task."""
    global _prime_status

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("missing_api_key", component="prime", event_msg="ANTHROPIC_API_KEY not set, generation loop disabled")
        return

    try:
        from src.loop import generation_loop
        await generation_loop()
    except Exception as e:
        log.error("generation_loop_error", component="prime", error=str(e))


async def main() -> None:
    """Main entry point."""
    global _prime_generation, _prime_status

    port = 8401
    log.info("prime_starting", component="prime", generation=_prime_generation, port=port)

    app = make_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=_prime_generation, port=port)

    # Run generation loop as background task
    loop_task = asyncio.create_task(run_generation_loop())

    try:
        # Keep server running
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
