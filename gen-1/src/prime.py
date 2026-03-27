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

from . import generate, manifest, supervisor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENERATION = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
SPEC_PATH = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
WORKSPACE = os.environ.get("CAMBRIAN_WORKSPACE", "/workspace")
TOKEN_BUDGET = int(os.environ.get("CAMBRIAN_TOKEN_BUDGET", "0"))  # 0 = unlimited

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(component="prime")

# ---------------------------------------------------------------------------
# Mutable state
# ---------------------------------------------------------------------------

_status = "idle"
_start_time = time.monotonic()


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------

async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "generation": GENERATION,
        "status": _status,
        "uptime": int(time.monotonic() - _start_time),
    })


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/health", handle_health)
    app.router.add_get("/stats", handle_stats)
    return app


# ---------------------------------------------------------------------------
# Parse repair helper
# ---------------------------------------------------------------------------

async def _parse_with_repair(
    response: str, model: str
) -> tuple[dict[str, str] | None, int]:
    """Attempt parse_files; on ParseError retry with a repair prompt up to MAX_PARSE_RETRIES.

    Returns (files, extra_tokens). extra_tokens counts tokens used by repair calls.
    A successful parse after repair does NOT consume a generation retry.
    """
    extra_tokens = 0
    current = response
    for attempt in range(MAX_PARSE_RETRIES + 1):
        try:
            return generate.parse_files(current), extra_tokens
        except generate.ParseError as e:
            if attempt >= MAX_PARSE_RETRIES:
                log.error(
                    "parse_retries_exhausted",
                    error=str(e),
                    max_parse_retries=MAX_PARSE_RETRIES,
                )
                return None, extra_tokens
            log.warning(
                "parse_error_attempting_repair",
                error=str(e),
                attempt=attempt + 1,
                max_parse_retries=MAX_PARSE_RETRIES,
            )
            repair_prompt = generate.build_parse_repair_prompt(str(e), current)
            try:
                current, r_in, r_out = await generate.call_llm(
                    system=generate.SYSTEM_PROMPT,
                    user=repair_prompt,
                    model=model,
                )
                extra_tokens += r_in + r_out
            except Exception as repair_err:
                log.error("parse_repair_llm_failed", error=str(repair_err))
                return None, extra_tokens
    return None, extra_tokens


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

async def generation_loop() -> None:
    """Main generation loop — produce offspring, verify, decide."""
    global _status

    spec_path = Path(SPEC_PATH)
    if not spec_path.exists():
        log.warning("spec_not_found", path=SPEC_PATH)
        return

    spec_content = spec_path.read_text()
    spec_hash = manifest.compute_spec_hash(spec_path)

    retry_count = 0
    last_failure: dict[str, Any] | None = None
    last_failed_files: dict[str, str] = {}
    generations_produced = 0
    cumulative_tokens = 0

    while generations_produced < MAX_GENS:
        _status = "idle"

        # 1. Determine generation number
        records = await supervisor.get_versions()
        if records:
            highest = max(r.get("generation", 0) for r in records)
            gen_num = highest + 1
        else:
            gen_num = 1

        parent = GENERATION

        # Model escalation: Sonnet on first attempt, Opus on retries
        current_model = MODEL if retry_count == 0 else ESCALATION_MODEL
        log.info(
            "generation_start",
            generation=gen_num,
            parent=parent,
            retry=retry_count,
            model=current_model,
        )

        # 2. Build prompt
        _status = "generating"
        if last_failure and retry_count > 0:
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

        # 3. Call LLM
        try:
            response_text, input_tokens, output_tokens = await generate.call_llm(
                system=generate.SYSTEM_PROMPT,
                user=user_prompt,
                model=current_model,
            )
        except Exception as e:
            log.error("llm_call_failed", error=str(e))
            retry_count += 1
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES)
                break
            continue

        # Track token budget
        cumulative_tokens += input_tokens + output_tokens
        if TOKEN_BUDGET > 0 and cumulative_tokens > TOKEN_BUDGET:
            log.error("token_budget_exhausted", budget=TOKEN_BUDGET, used=cumulative_tokens)
            break

        # 4. Parse response (with repair loop)
        files, repair_tokens = await _parse_with_repair(response_text, current_model)
        cumulative_tokens += repair_tokens
        if files is None:
            log.error("no_files_parsed", response_length=len(response_text))
            retry_count += 1
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES)
                break
            continue

        # 5. Write files to workspace
        artifact_dir = Path(WORKSPACE) / f"gen-{gen_num}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        file_list: list[str] = []

        for file_path, content in files.items():
            full_path = artifact_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            file_list.append(file_path)

        # 6. Copy spec into artifact
        spec_dest = artifact_dir / "spec" / "CAMBRIAN-SPEC-005.md"
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec_path, spec_dest)
        file_list.append("spec/CAMBRIAN-SPEC-005.md")

        # 7. Build and write manifest
        manifest_data = manifest.build_manifest(
            generation=gen_num,
            parent_generation=parent,
            spec_hash=spec_hash,
            artifact_dir=artifact_dir,
            files=file_list,
            producer_model=current_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            spec_content=spec_content,
        )
        manifest.write_manifest(artifact_dir, manifest_data)
        log.info("artifact_written", generation=gen_num, files=len(file_list))

        # 8. Request verification (relative path)
        _status = "verifying"
        spawn_result = await supervisor.spawn(
            generation=gen_num,
            artifact_path=f"gen-{gen_num}",
            spec_hash=spec_hash,
        )
        if not spawn_result.get("ok"):
            log.error("spawn_failed", error=spawn_result.get("error"))
            retry_count += 1
            if retry_count > MAX_RETRIES:
                break
            continue

        # 9. Poll until test rig completes
        record = await supervisor.poll_until_terminal(gen_num)
        viability = record.get("viability") or {}
        viable = viability.get("status") == "viable"
        log.info("verification_complete", generation=gen_num, viable=viable)

        # 10. Decide — Prime owns the promote/rollback decision
        if viable:
            promote_result = await supervisor.promote(gen_num)
            if promote_result.get("ok"):
                log.info("generation_promoted", generation=gen_num)
                retry_count = 0
                last_failure = None
                last_failed_files = {}
                generations_produced += 1
            else:
                log.error("promote_failed", generation=gen_num, error=promote_result.get("error"))
                break
        else:
            rollback_result = await supervisor.rollback(gen_num)
            if not rollback_result.get("ok"):
                log.warning("rollback_failed", generation=gen_num, error=rollback_result.get("error"))

            # Collect failure context for informed retry
            last_failure = viability.get("diagnostics", {})
            last_failed_files = _read_failed_files(artifact_dir, file_list)
            retry_count += 1
            log.warning(
                "generation_failed",
                generation=gen_num,
                retry_count=retry_count,
                max_retries=MAX_RETRIES,
            )
            if retry_count > MAX_RETRIES:
                log.error("max_retries_exhausted", retries=MAX_RETRIES)
                break

    _status = "idle"
    log.info("generation_loop_done", generations_produced=generations_produced,
             cumulative_tokens=cumulative_tokens)


def _read_failed_files(artifact_dir: Path, file_list: list[str]) -> dict[str, str]:
    """Read back files from a failed artifact for informed retry."""
    result: dict[str, str] = {}
    for f in file_list:
        p = artifact_dir / f
        if p.exists() and p.suffix in (".py", ".txt", ".md", ".json", ".toml", ".ini"):
            try:
                result[f] = p.read_text()
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """Start HTTP server, then run generation loop."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Fatal: ANTHROPIC_API_KEY is required but not set.")

    app = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8401)
    await site.start()
    log.info("prime_started", port=8401, generation=GENERATION)

    try:
        await generation_loop()
    except Exception as e:
        log.error("generation_loop_error", error=str(e))
    finally:
        # Keep serving /health until terminated
        log.info("generation_loop_ended_serving_health")
        try:
            await asyncio.Event().wait()  # block forever
        except asyncio.CancelledError:
            pass
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
