"""Gen-0 test artifact — hand-crafted minimal HTTP server for Phase 0 validation."""
import asyncio
import json
import time

from aiohttp import web

START_TIME = time.monotonic()


async def health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def stats(request: web.Request) -> web.Response:
    return web.json_response({
        "generation": 0,
        "status": "healthy",
        "uptime": round(time.monotonic() - START_TIME, 2),
    })


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/stats", stats)
    return app


if __name__ == "__main__":
    web.run_app(make_app(), host="0.0.0.0", port=8401)
