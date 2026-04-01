#!/usr/bin/env python3
"""Prime — the organism. Entry point, HTTP server, and main generation loop."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import structlog
from aiohttp import web

logger = structlog.get_logger()

# Global state
_start_time: float = time.time()
_generation: int = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
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
    """GET /stats — current status."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _generation,
        "status": _status,
        "uptime": uptime,
    })


async def run_generation_loop() -> None:
    """Background task: run the generation loop."""
    global _status

    from src.generate import (
        CAMBRIAN_MAX_GENS,
        CAMBRIAN_MAX_RETRIES,
        build_fresh_prompt,
        build_informed_retry_prompt,
        call_llm,
        get_generation_number,
        parse_files_with_repair,
        select_model,
    )
    from src.manifest import build_manifest, compute_spec_hash, write_manifest
    from src.supervisor import SupervisorClient

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set — generation loop disabled", component="prime")
        return

    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    spec_path = Path(spec_path_str)
    workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")

    supervisor = SupervisorClient(supervisor_url)

    log = logger.bind(component="prime")

    consecutive_failures = 0
    total_gens = 0
    retry_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while total_gens < CAMBRIAN_MAX_GENS and consecutive_failures < CAMBRIAN_MAX_RETRIES:
        _status = "generating"

        # Step 1: Determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as e:
            log.error("Failed to get versions from supervisor", error=str(e))
            await asyncio.sleep(5)
            continue

        offspring_gen = get_generation_number(versions)
        parent_gen = offspring_gen - 1

        # Step 2: Read spec
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
            spec_hash = compute_spec_hash(spec_path)
        except Exception as e:
            log.error("Failed to read spec", error=str(e), path=str(spec_path))
            await asyncio.sleep(5)
            continue

        log.info("Starting generation", generation=offspring_gen, retry_count=retry_count)

        # Step 3: Build prompt
        model = select_model(retry_count)
        if failed_artifact_path and failed_diagnostics:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path.exists():
                for fp in failed_artifact_path.rglob("*"):
                    if fp.is_file() and fp.name != "manifest.json":
                        rel = str(fp.relative_to(failed_artifact_path))
                        try:
                            failed_files[rel] = fp.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
            system_msg, user_msg = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=versions,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
                diagnostics=failed_diagnostics,
                failed_files=failed_files,
            )
        else:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=versions,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
            )

        # Step 4: Call LLM
        try:
            raw_response, token_usage = await call_llm(
                system_message=system_msg,
                user_message=user_msg,
                model=model,
            )
        except Exception as e:
            log.error("LLM call failed", error=str(e))
            await asyncio.sleep(5)
            continue

        # Step 5: Parse response
        try:
            files = await parse_files_with_repair(
                raw_response=raw_response,
                model=model,
            )
        except Exception as e:
            log.error("Parse failed after all repairs", error=str(e))
            consecutive_failures += 1
            total_gens += 1
            retry_count += 1
            failed_artifact_path = None
            failed_diagnostics = None
            continue

        # Step 6: Write files to workspace
        artifact_dir = workspace / f"gen-{offspring_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Build and write manifest
        all_files = list(files.keys()) + [f"spec/{spec_path.name}"]
        manifest = build_manifest(
            artifact_root=artifact_dir,
            files=all_files,
            generation=offspring_gen,
            parent_generation=parent_gen,
            spec_path=spec_path,
            model=model,
            token_usage=token_usage,
        )
        write_manifest(artifact_dir, manifest)

        log.info("Artifact written", generation=offspring_gen, path=str(artifact_dir))

        # Step 7: POST /spawn
        _status = "verifying"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=offspring_gen,
                artifact_path=f"gen-{offspring_gen}",
            )
            log.info("Spawn requested", generation=offspring_gen, result=spawn_result)
        except Exception as e:
            log.error("Spawn failed", error=str(e))
            await asyncio.sleep(5)
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("Polling for viability result", generation=offspring_gen)
        record = None
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
            except Exception as e:
                log.warning("Poll failed", error=str(e))
                continue

            for r in versions:
                if r.get("generation") == offspring_gen:
                    record = r
                    break

            if record and record.get("outcome") != "in_progress":
                break

        viability = record.get("viability", {}) if record else {}
        status = viability.get("status", "non-viable")

        total_gens += 1

        # Step 9: Decide
        if status == "viable":
            log.info("Viable! Promoting", generation=offspring_gen)
            try:
                await supervisor.promote(offspring_gen)
            except Exception as e:
                log.error("Promote failed", error=str(e))
            consecutive_failures = 0
            retry_count = 0
            failed_artifact_path = None
            failed_diagnostics = None
            _status = "idle"
            # One successful promotion per loop run
            break
        else:
            log.warning("Non-viable. Rolling back", generation=offspring_gen)
            try:
                await supervisor.rollback(offspring_gen)
            except Exception as e:
                log.error("Rollback failed", error=str(e))
            consecutive_failures += 1
            retry_count += 1
            failed_artifact_path = artifact_dir
            failed_diagnostics = viability.get("diagnostics")

    _status = "idle"
    log.info(
        "Generation loop finished",
        total_gens=total_gens,
        consecutive_failures=consecutive_failures,
    )


async def main() -> None:
    """Main entry point."""
    global _status

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )

    log = logger.bind(component="prime", generation=_generation)

    # Validate API key
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — generation loop will be disabled")

    log.info("Prime starting", generation=_generation)

    # Start HTTP server
    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()

    log.info("HTTP server started on port 8401")

    # Start generation loop as background task
    if api_key:
        asyncio.create_task(run_generation_loop())
    else:
        log.info("Generation loop disabled (no API key)")

    # Keep running
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("Shutting down")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
