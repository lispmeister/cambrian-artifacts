"""Generation loop — orchestrates LLM calls, file writing, and Supervisor interaction."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

WORKSPACE = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
ARTIFACTS_ROOT = Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace"))
SPEC_PATH = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
SUPERVISOR_URL = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
    from src.supervisor import SupervisorClient
    from src.generate import (
        build_fresh_prompt,
        build_retry_prompt,
        build_repair_prompt,
        call_llm,
        parse_files,
        ParseError,
        get_model,
        SYSTEM_PROMPT,
    )
    from src.manifest import build_manifest, write_manifest, compute_spec_hash

    supervisor = SupervisorClient(SUPERVISOR_URL)

    total_gens = 0
    consecutive_failures = 0

    while total_gens < MAX_GENS and consecutive_failures < MAX_RETRIES:
        prime_module._status = "generating"

        # Step 1: Determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as e:
            log.error("supervisor_error", component="prime", error=str(e), step="get_versions")
            await _backoff_sleep(consecutive_failures)
            continue

        if versions:
            next_gen = max(v.get("generation", 0) for v in versions) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        log.info("generation_starting", component="prime", generation=next_gen, parent=parent_gen)

        # Step 2: Read spec
        try:
            spec_content = SPEC_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.error("spec_not_found", component="prime", path=str(SPEC_PATH))
            return

        spec_hash = compute_spec_hash(SPEC_PATH)

        # Step 3 & 4: Build prompt and call LLM (with parse repair loop)
        history_json = json.dumps(versions, indent=2)

        # Determine if this is a retry (consecutive_failures > 0)
        retry_count = consecutive_failures
        model = get_model(retry_count)

        if retry_count == 0:
            user_msg = build_fresh_prompt(spec_content, history_json, next_gen, parent_gen)
        else:
            # Load failed code from previous attempt
            prev_gen = next_gen - 1
            failed_dir = ARTIFACTS_ROOT / f"gen-{prev_gen}"
            failed_files: dict[str, str] = {}
            if failed_dir.exists():
                for fp in failed_dir.rglob("*"):
                    if fp.is_file():
                        rel = str(fp.relative_to(failed_dir))
                        try:
                            failed_files[rel] = fp.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass

            # Get diagnostics from last version record
            diagnostics: dict[str, Any] = {}
            if versions:
                last = versions[-1]
                viability = last.get("viability", {})
                if viability:
                    diagnostics = viability.get("diagnostics", {})

            user_msg = build_retry_prompt(
                spec_content, history_json, next_gen, parent_gen,
                prev_gen, diagnostics, failed_files
            )

        # Parse loop with repair
        parsed_files: dict[str, str] | None = None
        parse_failures = 0

        for attempt in range(MAX_PARSE_RETRIES + 1):
            try:
                prime_module._status = "generating"
                raw_response = await call_llm(model, SYSTEM_PROMPT, user_msg)
                parsed_files = parse_files(raw_response)
                break
            except ParseError as pe:
                parse_failures += 1
                log.warning("parse_error", component="prime", error=str(pe), attempt=attempt)
                if attempt < MAX_PARSE_RETRIES:
                    user_msg = build_repair_prompt(str(pe), raw_response)
                else:
                    log.error("parse_repair_exhausted", component="prime", generation=next_gen)
                    break
            except Exception as e:
                log.error("llm_error", component="prime", error=str(e))
                await asyncio.sleep(5)
                break

        if parsed_files is None:
            consecutive_failures += 1
            total_gens += 1
            log.error("generation_failed_parse", component="prime", generation=next_gen)
            continue

        # Step 6: Write files to workspace
        artifact_dir = ARTIFACTS_ROOT / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file into artifact
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / SPEC_PATH.name
        shutil.copy2(SPEC_PATH, spec_dest)

        # Build file list
        file_list: list[str] = []
        for fp in sorted(artifact_dir.rglob("*")):
            if fp.is_file():
                rel = str(fp.relative_to(artifact_dir))
                file_list.append(rel)
        if "manifest.json" not in file_list:
            file_list.append("manifest.json")

        # Extract token usage from last LLM call (stored globally by call_llm)
        from src import generate as gen_module
        token_usage = getattr(gen_module, "_last_token_usage", {"input": 0, "output": 0})

        # Build and write manifest
        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_path=SPEC_PATH,
            artifact_root=artifact_dir,
            files=file_list,
            model=model,
            token_usage=token_usage,
            spec_content=spec_content,
        )
        write_manifest(artifact_dir, manifest)

        log.info("artifact_written", component="prime", generation=next_gen, path=str(artifact_dir))

        # Step 7: POST /spawn
        prime_module._status = "verifying"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn_requested", component="prime", generation=next_gen, result=spawn_result)
        except Exception as e:
            log.error("spawn_error", component="prime", error=str(e))
            consecutive_failures += 1
            total_gens += 1
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("polling_for_outcome", component="prime", generation=next_gen)
        viability_status = None
        viability_diagnostics: dict[str, Any] = {}

        for _ in range(600):  # max 20 minutes
            await asyncio.sleep(2)
            try:
                versions_now = await supervisor.get_versions()
            except Exception as e:
                log.warning("poll_error", component="prime", error=str(e))
                continue

            record = next(
                (v for v in versions_now if v.get("generation") == next_gen),
                None
            )
            if record is None:
                continue

            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                viability = record.get("viability", {})
                viability_status = viability.get("status") if viability else None
                viability_diagnostics = viability.get("diagnostics", {}) if viability else {}
                log.info("outcome_received", component="prime", generation=next_gen,
                         outcome=outcome, viability=viability_status)
                break

        if viability_status is None:
            log.error("poll_timeout", component="prime", generation=next_gen)
            consecutive_failures += 1
            total_gens += 1
            continue

        # Step 9: Decide
        total_gens += 1

        if viability_status == "viable":
            try:
                await supervisor.promote(next_gen)
                log.info("generation_promoted", component="prime", generation=next_gen)
                consecutive_failures = 0
                # Stop after one successful promotion per loop run
                break
            except Exception as e:
                log.error("promote_error", component="prime", error=str(e))
        else:
            try:
                await supervisor.rollback(next_gen)
                log.info("generation_rolled_back", component="prime", generation=next_gen)
            except Exception as e:
                log.error("rollback_error", component="prime", error=str(e))
            consecutive_failures += 1
            log.warning("generation_failed", component="prime", generation=next_gen,
                        consecutive_failures=consecutive_failures)

    prime_module._status = "idle"
    log.info("generation_loop_complete", component="prime",
             total_gens=total_gens, consecutive_failures=consecutive_failures)


async def _backoff_sleep(attempt: int) -> None:
    """Exponential backoff sleep."""
    delay = min(2 ** attempt, 60)
    await asyncio.sleep(delay)
