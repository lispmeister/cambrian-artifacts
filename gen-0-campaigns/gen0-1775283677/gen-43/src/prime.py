"""Prime — the organism. Entry point, HTTP server, main loop."""
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
    """GET /stats — status endpoint."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _prime_status,
        "uptime": uptime,
    })


def set_status(status: str) -> None:
    """Update the prime status."""
    global _prime_status
    _prime_status = status


async def run_generation_loop() -> None:
    """Run the generation loop as a background task."""
    global _prime_status

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("anthropic_key_missing", component="prime", event="anthropic_key_missing")
        return

    try:
        from src.loop import generation_loop
        await generation_loop()
    except Exception as exc:
        log.error("generation_loop_error", component="prime", error=str(exc))
    finally:
        _prime_status = "idle"


async def main() -> None:
    """Main entry point."""
    generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    port = 8401

    log.info("prime_starting", component="prime", generation=generation, port=port)

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    log.info("prime_listening", component="prime", generation=generation, port=port)

    loop_task = asyncio.create_task(run_generation_loop())

    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
