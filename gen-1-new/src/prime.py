"""Prime — the organism. Async HTTP server + generation loop."""
from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any

import structlog
from aiohttp import web

# Support both `python src/prime.py` (direct) and `python -m src.prime` (module).
# When run directly, __package__ is None so relative imports fail.
try:
    from . import generate, manifest, supervisor
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src import generate, manifest, supervisor  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(component="prime")

# ---------------------------------------------------------------------------
# Configuration (read at import time; overridable via env)
# ---------------------------------------------------------------------------

MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
SPEC_PATH = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
WORKSPACE = os.environ.get("CAMBRIAN_WORKSPACE", "/workspace")
TOKEN_BUDGET = int(os.environ.get("CAMBRIAN_TOKEN_BUDGET", "0"))  # 0 = unlimited
PORT = int(os.environ.get("CAMBRIAN_PORT", "8401"))

# ---------------------------------------------------------------------------
# Mutable server state
# ---------------------------------------------------------------------------

_status = "idle"           # "idle" | "generating" | "verifying"
# Own generation identity — from CAMBRIAN_GENERATION env var (Supervisor sets this).
# Stays fixed; does not change as the loop produces offspring.
_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
_start_time = time.monotonic()


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_health(request: web.Request) -> web.Response:
    """Liveness check. Always returns 200 {"ok": true}."""
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    """Stats: generation number, status, uptime in seconds."""
    return web.json_response({
        "generation": _generation,
        "status": _status,
        "uptime": int(time.monotonic() - _start_time),
    })


async def handle_versions(request: web.Request) -> web.Response:
    """Proxy GET /versions to the Supervisor and return the array as-is."""
    versions = await supervisor.get_versions()
    return web.json_response(versions)


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    app.router.add_get("/versions", handle_versions)
    return app


# ---------------------------------------------------------------------------
# Generation loop helpers
# ---------------------------------------------------------------------------

def _read_failed_files(artifact_dir: Path, file_list: list[str]) -> dict[str, str]:
    """Read back source files from a failed artifact for informed retry.

    Only reads text-like files to avoid binary blobs.
    """
    result: dict[str, str] = {}
    text_suffixes = {".py", ".txt", ".md", ".json", ".toml", ".ini", ".cfg", ".yaml", ".yml"}
    for rel in file_list:
        p = artifact_dir / rel
        if p.exists() and p.suffix in text_suffixes:
            try:
                result[rel] = p.read_text()
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

async def generation_loop() -> None:
    """Main generation loop — produce offspring, verify, decide."""
    global _status, _generation

    spec_path = Path(SPEC_PATH)
    if not spec_path.exists():
        log.warning("spec_not_found", path=SPEC_PATH)
        return

    spec_content = spec_path.read_text()
    spec_hash = manifest.compute_spec_hash(spec_path)
    log.info("spec_loaded", path=SPEC_PATH, spec_hash=spec_hash)

    retry_count = 0
    last_failure: dict[str, Any] | None = None
    last_failed_files: dict[str, str] = {}
    last_failed_artifact_dir: Path | None = None
    generations_produced = 0
    cumulative_tokens = 0

    while generations_produced < MAX_GENS:
        _status = "idle"

        # Step 1: Determine generation number
        records = await supervisor.get_versions()
        if records:
            highest = max(r.get("generation", 0) for r in records)
            gen_num = highest + 1
        else:
            gen_num = 1

        # _generation tracks Prime's own identity (set at startup from CAMBRIAN_GENERATION).
        # gen_num is the offspring being produced — not exposed via /stats.

        # Parent is the last promoted generation (0 for bootstrap)
        parent = 0
        promoted = [r for r in records if r.get("outcome") == "promoted"]
        if promoted:
            parent = max(r.get("generation", 0) for r in promoted)

        log.info("generation_start", generation=gen_num, parent=parent, retry=retry_count)

        # Step 3: Build prompt (step 2 is spec read — done above)
        _status = "generating"
        if last_failure is not None and retry_count > 0:
            user_prompt = generate.build_retry_prompt(
                spec_content=spec_content,
                generation_records=records,
                generation=gen_num,
                parent=parent,
                failed_generation=gen_num - 1,
                diagnostics=last_failure,
                failed_files=last_failed_files,
            )
        else:
            user_prompt = generate.build_fresh_prompt(
                spec_content=spec_content,
                generation_records=records,
                generation=gen_num,
                parent=parent,
            )

        # Step 4: Call LLM
        try:
            response_text, input_tokens, output_tokens = await generate.call_llm(
                system=generate.SYSTEM_PROMPT,
                user=user_prompt,
                model=MODEL,
            )
        except Exception as e:
            log.error("llm_call_failed", error=str(e), generation=gen_num)
            retry_count += 1
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES)
                break
            continue

        # Track token budget
        cumulative_tokens += input_tokens + output_tokens
        if TOKEN_BUDGET > 0 and cumulative_tokens > TOKEN_BUDGET:
            log.error(
                "token_budget_exhausted",
                budget=TOKEN_BUDGET,
                used=cumulative_tokens,
                generation=gen_num,
            )
            break

        # Step 5: Parse response
        files = generate.parse_files(response_text)
        if not files:
            log.error("no_files_parsed", response_length=len(response_text), generation=gen_num)
            retry_count += 1
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES)
                break
            continue

        # Step 6: Write files to workspace
        artifact_dir = Path(WORKSPACE) / f"gen-{gen_num}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        file_list: list[str] = []

        for file_path, content in files.items():
            full_path = artifact_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            file_list.append(file_path)

        # Copy spec into artifact (faithful inheritance — spec is the genome)
        spec_dest = artifact_dir / "spec" / "CAMBRIAN-SPEC-005.md"
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec_path, spec_dest)
        spec_rel = "spec/CAMBRIAN-SPEC-005.md"
        if spec_rel not in file_list:
            file_list.append(spec_rel)

        # Build and write manifest (computed, not LLM-generated)
        manifest_data = manifest.build_manifest(
            generation=gen_num,
            parent_generation=parent,
            spec_hash=spec_hash,
            artifact_dir=artifact_dir,
            files=file_list,
            producer_model=MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        manifest.write_manifest(artifact_dir, manifest_data)
        log.info("artifact_written", generation=gen_num, files=len(file_list))

        # Step 7: Request verification (relative artifact path — Supervisor resolves host path)
        _status = "verifying"
        spawn_result = await supervisor.spawn(
            generation=gen_num,
            artifact_path=f"gen-{gen_num}",
            spec_hash=spec_hash,
        )
        if not spawn_result.get("ok"):
            log.error("spawn_failed", error=spawn_result.get("error"), generation=gen_num)
            retry_count += 1
            if retry_count > MAX_RETRIES:
                break
            continue

        # Step 8: Poll until outcome != in_progress
        record = await supervisor.poll_until_tested(gen_num)
        viability = record.get("viability") or {}
        viable = viability.get("status") == "viable"
        log.info("verification_complete", generation=gen_num, viable=viable)

        # Step 9: Decide
        if viable:
            promote_result = await supervisor.promote(gen_num)
            if promote_result.get("ok"):
                log.info("generation_promoted", generation=gen_num)
                retry_count = 0
                last_failure = None
                last_failed_files = {}
                last_failed_artifact_dir = None
                generations_produced += 1
            else:
                log.error(
                    "promote_failed",
                    generation=gen_num,
                    error=promote_result.get("error"),
                )
                break
        else:
            rollback_result = await supervisor.rollback(gen_num)
            if not rollback_result.get("ok"):
                log.warning(
                    "rollback_failed",
                    generation=gen_num,
                    error=rollback_result.get("error"),
                )

            # Collect failure context for informed retry
            diagnostics = viability.get("diagnostics", {})
            last_failure = diagnostics if diagnostics else {"stage": "unknown", "summary": "non-viable"}
            last_failed_files = _read_failed_files(artifact_dir, file_list)
            last_failed_artifact_dir = artifact_dir
            retry_count += 1

            log.warning(
                "generation_failed",
                generation=gen_num,
                retry_count=retry_count,
                max_retries=MAX_RETRIES,
            )
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES, generation=gen_num)
                break

    _status = "idle"
    log.info(
        "generation_loop_done",
        generations_produced=generations_produced,
        cumulative_tokens=cumulative_tokens,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Validate env, start HTTP server, then run generation loop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Fatal: ANTHROPIC_API_KEY is required but not set.")

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    log.info("prime_started", port=PORT, model=MODEL, workspace=WORKSPACE)

    try:
        await generation_loop()
    except Exception as e:
        log.error("generation_loop_error", error=str(e))
    finally:
        # Keep serving /health after loop ends (Test Rig may still be polling)
        log.info("generation_loop_ended_serving_health")
        try:
            await asyncio.Event().wait()  # block until cancelled
        except asyncio.CancelledError:
            pass
        await supervisor.close()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
