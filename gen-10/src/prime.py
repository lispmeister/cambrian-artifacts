#!/usr/bin/env python3
"""Prime — the organism. Async HTTP server + generation loop."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import structlog
from aiohttp import web

from src.generate import GenerationConfig, LLMGenerator
from src.manifest import build_manifest, compute_artifact_hash, compute_spec_hash, write_manifest
from src.models import GenerationRecord, ViabilityReport
from src.supervisor import SupervisorClient

logger = structlog.get_logger().bind(component="prime")

_start_time = time.time()
_status = "idle"
_generation_number = int(os.environ.get("CAMBRIAN_GENERATION", "0"))


def make_app() -> web.Application:
    """Create and return the aiohttp application."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/stats", stats_handler)
    return app


async def health_handler(request: web.Request) -> web.Response:
    """GET /health — liveness check."""
    return web.json_response({"ok": True})


async def stats_handler(request: web.Request) -> web.Response:
    """GET /stats — status information."""
    uptime = int(time.time() - _start_time)
    return web.json_response({
        "generation": _generation_number,
        "status": _status,
        "uptime": uptime,
    })


def set_status(new_status: str) -> None:
    """Update the global status."""
    global _status
    _status = new_status


async def generation_loop(config: GenerationConfig, supervisor: SupervisorClient) -> None:
    """Main generation loop — produces offspring generations."""
    global _status

    log = logger.bind(task="generation_loop")
    log.info("Starting generation loop")

    retry_count = 0
    max_retries = config.max_retries
    max_gens = config.max_gens
    gens_attempted = 0
    failed_context: dict[str, Any] | None = None

    while gens_attempted < max_gens and retry_count <= max_retries:
        set_status("generating")

        # Step 1: Determine generation number
        try:
            history = await supervisor.get_versions()
        except Exception as e:
            log.error("Failed to get versions from supervisor", error=str(e))
            await asyncio.sleep(5)
            continue

        if history:
            next_gen = max(r.generation for r in history) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1
        log.info("Determined generation number", next_gen=next_gen, parent_gen=parent_gen)

        # Step 2: Read spec
        spec_path = config.spec_path
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except Exception as e:
            log.error("Failed to read spec file", path=str(spec_path), error=str(e))
            await asyncio.sleep(5)
            continue

        spec_hash = compute_spec_hash(spec_path)

        # Step 3 & 4: Build prompt and call LLM
        generator = LLMGenerator(config)
        history_data = [r.model_dump() for r in history]

        parse_retries = 0
        files: dict[str, str] | None = None
        raw_response: str | None = None
        token_usage: dict[str, int] = {"input": 0, "output": 0}
        model_used: str = config.model if retry_count == 0 else config.escalation_model

        while parse_retries <= config.max_parse_retries:
            try:
                if parse_retries == 0:
                    result = await generator.generate(
                        spec_content=spec_content,
                        history=history_data,
                        generation=next_gen,
                        parent=parent_gen,
                        retry_count=retry_count,
                        failed_context=failed_context,
                    )
                else:
                    assert raw_response is not None
                    assert last_parse_error is not None
                    result = await generator.repair(
                        raw_response=raw_response,
                        parse_error=last_parse_error,
                        retry_count=retry_count,
                    )

                files = result["files"]
                raw_response = result["raw_response"]
                token_usage = result["token_usage"]
                model_used = result["model"]
                break

            except Exception as e:
                last_parse_error = str(e)
                log.warning("Parse error, attempting repair", attempt=parse_retries, error=str(e))
                parse_retries += 1
                if parse_retries > config.max_parse_retries:
                    log.error("All parse repairs failed", generation=next_gen)
                    files = None
                    break

        if files is None:
            log.error("Failed to generate files after parse retries", generation=next_gen)
            retry_count += 1
            gens_attempted += 1
            failed_context = None
            continue

        # Step 6: Write files to workspace
        set_status("verifying")
        workspace_root = config.workspace_root
        artifact_dir = workspace_root / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            file_path = artifact_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            log.debug("Wrote file", path=rel_path)

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        spec_rel_path = f"spec/{spec_path.name}"
        all_files = list(files.keys()) + [spec_rel_path]

        # Build and write manifest
        artifact_hash = compute_artifact_hash(artifact_dir, all_files)
        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_hash=artifact_hash,
            producer_model=model_used,
            token_usage=token_usage,
            files=all_files,
            spec_content=spec_content,
        )
        write_manifest(artifact_dir, manifest)

        log.info("Artifact written", generation=next_gen, artifact_dir=str(artifact_dir))

        # Step 7: POST /spawn
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("Spawn requested", generation=next_gen, result=spawn_result)
        except Exception as e:
            log.error("Failed to spawn", generation=next_gen, error=str(e))
            retry_count += 1
            gens_attempted += 1
            failed_context = None
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("Polling for viability result", generation=next_gen)
        viability: ViabilityReport | None = None
        outcome: str = "in_progress"

        while True:
            await asyncio.sleep(2)
            try:
                current_history = await supervisor.get_versions()
            except Exception as e:
                log.warning("Poll error", error=str(e))
                continue

            target_record: GenerationRecord | None = None
            for record in current_history:
                if record.generation == next_gen:
                    target_record = record
                    break

            if target_record is None:
                log.warning("Generation record not found yet", generation=next_gen)
                continue

            outcome = target_record.outcome
            if outcome != "in_progress":
                viability = target_record.viability
                log.info("Generation outcome determined", generation=next_gen, outcome=outcome)
                break

        # Step 9: Decide
        gens_attempted += 1

        if viability is not None and viability.status == "viable":
            try:
                await supervisor.promote(generation=next_gen)
                log.info("Generation promoted", generation=next_gen)
            except Exception as e:
                log.error("Failed to promote", generation=next_gen, error=str(e))
            retry_count = 0
            failed_context = None
            set_status("idle")

        else:
            # Non-viable
            try:
                await supervisor.rollback(generation=next_gen)
                log.info("Generation rolled back", generation=next_gen)
            except Exception as e:
                log.error("Failed to rollback", generation=next_gen, error=str(e))

            retry_count += 1
            if retry_count > max_retries:
                log.error("Max retries exhausted", generation=next_gen)
                set_status("idle")
                break

            # Build failed context for informed retry
            if viability is not None and viability.diagnostics is not None:
                failed_files: dict[str, str] = {}
                for rel_path in files.keys():
                    fp = artifact_dir / rel_path
                    if fp.exists():
                        try:
                            failed_files[rel_path] = fp.read_text(encoding="utf-8")
                        except Exception:
                            pass
                failed_context = {
                    "generation": next_gen,
                    "diagnostics": viability.diagnostics.model_dump(),
                    "files": failed_files,
                }
            else:
                failed_context = None

    log.info("Generation loop ended", gens_attempted=gens_attempted, retry_count=retry_count)
    set_status("idle")


async def main() -> None:
    """Entry point — start HTTP server then generation loop."""
    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )

    # Validate required env vars
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set — fatal error")
        raise SystemExit(1)

    from pathlib import Path
    config = GenerationConfig(
        anthropic_api_key=api_key,
        model=os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6"),
        escalation_model=os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6"),
        max_gens=int(os.environ.get("CAMBRIAN_MAX_GENS", "5")),
        max_retries=int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3")),
        max_parse_retries=int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2")),
        spec_path=Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")),
        workspace_root=Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace")),
    )

    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
    supervisor = SupervisorClient(base_url=supervisor_url)

    # Start HTTP server first (before generation loop)
    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()
    logger.info("HTTP server started", port=8401)

    # Start generation loop as background task
    loop_task = asyncio.create_task(generation_loop(config, supervisor))

    try:
        await loop_task
    except asyncio.CancelledError:
        logger.info("Generation loop cancelled")
    finally:
        await runner.cleanup()
        await supervisor.close()


if __name__ == "__main__":
    asyncio.run(main())