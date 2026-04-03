"""Prime — the organism. Entry point, HTTP server, main loop."""
from __future__ import annotations

import asyncio
import os
import time

import structlog
from aiohttp import web

from src.loop import generation_loop
from src.supervisor import SupervisorClient

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

logger = structlog.get_logger()

# Global state
_start_time: float = time.time()
_prime_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_status_holder: dict[str, str] = {"status": "idle"}


async def health_handler(request: web.Request) -> web.Response:
    """GET /health - liveness check."""
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    """GET /stats - status information."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _prime_generation,
        "status": _status_holder.get("status", "idle"),
        "uptime": uptime,
    })


def make_app() -> web.Application:
    """Create the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def run_generation_loop() -> None:
    """Run the generation loop as a background task."""
    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )
    supervisor = SupervisorClient(supervisor_url)
    try:
        await generation_loop(supervisor, _status_holder)
    except Exception as exc:
        logger.error(
            "generation_loop_error",
            component="prime",
            generation=_prime_generation,
            error=str(exc),
        )
    finally:
        await supervisor.close()


async def main() -> None:
    """Main entry point."""
    global _start_time, _prime_generation

    _start_time = time.time()
    _prime_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))

    port = 8401

    logger.info(
        "prime_starting",
        component="prime",
        generation=_prime_generation,
        port=port,
    )

    # Start HTTP server first, then generation loop
    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()

    logger.info(
        "prime_listening",
        component="prime",
        generation=_prime_generation,
        port=port,
    )

    # Start generation loop as background task
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        loop_task = asyncio.create_task(run_generation_loop())
    else:
        logger.warning(
            "no_api_key_skipping_generation",
            component="prime",
            generation=_prime_generation,
        )
        loop_task = None

    # Keep running until interrupted
    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        if loop_task and not loop_task.done():
            loop_task.cancel()
            try:
                await loop_task
            except asyncio.CancelledError:
                pass
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
