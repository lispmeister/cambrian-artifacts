"""Generation loop — orchestrates the full generate/verify/decide cycle."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

WORKSPACE_ROOT = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
ARTIFACTS_ROOT = Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace"))
SPEC_PATH = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
SUPERVISOR_URL = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
POLL_INTERVAL = 2.0


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
    from src.generate import (
        ParseError,
        build_fresh_prompt,
        build_informed_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        get_model,
        parse_files,
    )
    from src.manifest import build_manifest, compute_artifact_hash, compute_spec_hash, write_manifest
    from src.supervisor import SupervisorClient

    supervisor = SupervisorClient(SUPERVISOR_URL)
    gen_count = 0
    retry_count = 0

    while gen_count < MAX_GENS:
        # Step 1: determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as exc:
            log.error("supervisor_unreachable", component="prime", error=str(exc))
            await asyncio.sleep(5)
            continue

        if versions:
            next_gen = max(v.get("generation", 0) for v in versions) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        # Step 2: read spec
        spec_path = SPEC_PATH
        if not spec_path.is_absolute():
            spec_path = WORKSPACE_ROOT / spec_path
        try:
            spec_content = spec_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.error("spec_not_found", component="prime", path=str(spec_path))
            return

        spec_hash = compute_spec_hash(spec_path)
        log.info("spec_loaded", component="prime", generation=next_gen, spec_hash=spec_hash)

        # Steps 3-5: build prompt, call LLM, parse response
        prime_module._status = "generating"
        model = get_model(retry_count)

        # Collect failed files for informed retry
        failed_files: dict[str, str] = {}
        diagnostics: dict[str, Any] | None = None
        if retry_count > 0:
            # Load failed source from previous generation directory
            failed_gen = next_gen - 1
            failed_dir = ARTIFACTS_ROOT / f"gen-{failed_gen}"
            if failed_dir.exists():
                for fp in failed_dir.rglob("*"):
                    if fp.is_file() and fp.name != "manifest.json":
                        rel = str(fp.relative_to(failed_dir))
                        try:
                            failed_files[rel] = fp.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
            # Get diagnostics from generation record
            try:
                for rec in versions:
                    if rec.get("generation") == failed_gen:
                        viability = rec.get("viability", {})
                        diagnostics = viability.get("diagnostics")
                        break
            except Exception:
                pass

        # Parse repair loop
        raw_response: str | None = None
        files: dict[str, str] = {}
        parse_error_count = 0
        parse_success = False

        history_json = json.dumps(versions, indent=2)

        while not parse_success:
            try:
                if raw_response is None:
                    # Fresh call
                    if retry_count == 0 or not failed_files:
                        prompt = build_fresh_prompt(spec_content, history_json, next_gen, parent_gen)
                    else:
                        prompt = build_informed_retry_prompt(
                            spec_content, history_json, next_gen, parent_gen,
                            failed_gen, diagnostics or {}, failed_files
                        )
                    raw_response = await call_llm(model, prompt)
                else:
                    # Parse repair
                    repair_prompt = build_parse_repair_prompt(str(parse_error), raw_response)
                    raw_response = await call_llm(model, repair_prompt)

                files = parse_files(raw_response)
                parse_success = True

            except ParseError as exc:
                parse_error = exc
                parse_error_count += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    generation=next_gen,
                    attempt=parse_error_count,
                    error=str(exc),
                )
                if parse_error_count >= MAX_PARSE_RETRIES:
                    log.error("parse_repair_exhausted", component="prime", generation=next_gen)
                    break

        if not parse_success:
            # Count as generation failure
            gen_count += 1
            retry_count += 1
            prime_module._status = "idle"
            if retry_count >= MAX_RETRIES:
                log.error("max_retries_reached", component="prime")
                return
            continue

        # Step 6: write files
        artifact_dir = ARTIFACTS_ROOT / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = spec_path.name
        shutil.copy2(spec_path, spec_dest_dir / spec_filename)

        # Build file list
        all_files: list[str] = []
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                rel = str(fp.relative_to(artifact_dir))
                all_files.append(rel)

        # Compute artifact hash (before writing manifest)
        artifact_hash = compute_artifact_hash(artifact_dir, all_files + ["manifest.json"])

        # Build and write manifest
        # Extract token usage from last LLM call
        token_usage = {"input": 0, "output": 0}

        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_hash=artifact_hash,
            files=all_files + ["manifest.json"],
            token_usage=token_usage,
            spec_path=spec_path,
            spec_content=spec_content,
        )
        write_manifest(artifact_dir, manifest)

        log.info(
            "artifact_written",
            component="prime",
            generation=next_gen,
            artifact_dir=str(artifact_dir),
        )

        # Step 7: POST /spawn
        prime_module._status = "verifying"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawned", component="prime", generation=next_gen, result=spawn_result)
        except Exception as exc:
            log.error("spawn_failed", component="prime", generation=next_gen, error=str(exc))
            gen_count += 1
            retry_count += 1
            prime_module._status = "idle"
            continue

        # Step 8: poll until outcome != in_progress
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                versions = await supervisor.get_versions()
            except Exception as exc:
                log.warning("poll_error", component="prime", error=str(exc))
                continue

            record = None
            for v in versions:
                if v.get("generation") == next_gen:
                    record = v
                    break

            if record is None:
                continue

            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                break

        # Step 9: decide
        viability = record.get("viability", {})
        status = viability.get("status", "non-viable")

        gen_count += 1

        if status == "viable":
            log.info("promoting", component="prime", generation=next_gen)
            try:
                await supervisor.promote(next_gen)
            except Exception as exc:
                log.error("promote_failed", component="prime", error=str(exc))
            retry_count = 0
            prime_module._status = "idle"
            return  # Done after one promotion
        else:
            log.info("rolling_back", component="prime", generation=next_gen)
            try:
                await supervisor.rollback(next_gen)
            except Exception as exc:
                log.error("rollback_failed", component="prime", error=str(exc))
            retry_count += 1
            prime_module._status = "idle"
            if retry_count >= MAX_RETRIES:
                log.error("max_retries_reached", component="prime")
                return

    log.info("max_gens_reached", component="prime", gen_count=gen_count)
