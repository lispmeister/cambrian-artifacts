"""Minimal test artifact — a tiny aiohttp server that satisfies the Test Rig pipeline."""
import os

from aiohttp import web

PORT = int(os.environ.get("CAMBRIAN_PORT", "8401"))
_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "generation": _generation,
        "status": "idle",
        "uptime": 0,
    })


app = web.Application()
app.router.add_get("/health", handle_health)
app.router.add_get("/stats", handle_stats)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT)
