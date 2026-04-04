"""Generation loop — coordinates the LLM generation and verification cycle."""
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


async def generation_loop() -> None:
    """Main generation loop."""
    from src.supervisor import SupervisorClient
    from src.generate import (
        ParseError,
        build_fresh_prompt,
        build_informed_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        parse_files,
        get_model,
    )
    from src.manifest import build_manifest, write_manifest, compute_spec_hash

    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )
    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    workspace_root = Path("/workspace")
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))

    spec_path = Path(spec_path_str)
    if not spec_path.is_absolute():
        spec_path = workspace_root / spec_path_str

    supervisor = SupervisorClient(supervisor_url)

    # Read spec
    try:
        spec_content = spec_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.error("spec_not_found", component="prime", path=str(spec_path))
        return

    spec_hash = compute_spec_hash(spec_path)
    log.info("spec_loaded", component="prime", spec_hash=spec_hash)

    gen_count = 0
    retry_count = 0
    failure_context: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        # Step 1: Determine generation number
        try:
            history = await supervisor.get_versions()
        except Exception as exc:
            log.error("supervisor_unreachable", component="prime", error=str(exc))
            await asyncio.sleep(5)
            continue

        if history:
            offspring_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            offspring_gen = 1

        parent_gen = offspring_gen - 1
        log.info(
            "generation_starting",
            component="prime",
            offspring_gen=offspring_gen,
            retry_count=retry_count,
        )

        # Step 3: Build prompt
        if failure_context is not None:
            system_msg, user_msg = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=history,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
                failure_context=failure_context,
            )
        else:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=history,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
            )

        # Step 4: Call LLM with parse retry loop
        model = get_model(retry_count)
        raw_response: str | None = None
        files: dict[str, str] | None = None
        parse_repair_count = 0

        try:
            raw_response, token_usage = await call_llm(
                system_message=system_msg,
                user_message=user_msg,
                model=model,
            )
        except Exception as exc:
            log.error("llm_call_failed", component="prime", error=str(exc))
            retry_count += 1
            gen_count += 1
            failure_context = None
            continue

        # Step 5: Parse response
        while raw_response is not None:
            try:
                files = parse_files(raw_response)
                break
            except ParseError as exc:
                parse_repair_count += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    error=str(exc),
                    repair_attempt=parse_repair_count,
                )
                if parse_repair_count >= max_parse_retries:
                    log.error("parse_repair_exhausted", component="prime")
                    files = None
                    raw_response = None
                    break
                # Attempt parse repair
                repair_system, repair_user = build_parse_repair_prompt(
                    parse_error_message=str(exc),
                    raw_response=raw_response,
                )
                try:
                    raw_response, _ = await call_llm(
                        system_message=repair_system,
                        user_message=repair_user,
                        model=model,
                    )
                except Exception as repair_exc:
                    log.error("llm_repair_failed", component="prime", error=str(repair_exc))
                    raw_response = None
                    break

        if files is None:
            retry_count += 1
            gen_count += 1
            failure_context = None
            continue

        # Step 6: Write files
        artifact_dir = workspace_root / f"gen-{offspring_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest = artifact_dir / "spec" / spec_path.name
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec_path, spec_dest)

        file_list = [str(p.relative_to(artifact_dir)) for p in artifact_dir.rglob("*") if p.is_file()]
        file_list.sort()

        manifest_data = build_manifest(
            artifact_root=artifact_dir,
            files=file_list,
            generation=offspring_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            producer_model=model,
            token_usage=token_usage,
            spec_path=spec_path,
        )
        write_manifest(artifact_dir, manifest_data)

        log.info(
            "artifact_written",
            component="prime",
            generation=offspring_gen,
            artifact_dir=str(artifact_dir),
        )

        # Step 7: POST /spawn
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=offspring_gen,
                artifact_path=f"gen-{offspring_gen}",
            )
            log.info("spawn_requested", component="prime", result=spawn_result)
        except Exception as exc:
            log.error("spawn_failed", component="prime", error=str(exc))
            retry_count += 1
            gen_count += 1
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("polling_for_result", component="prime", generation=offspring_gen)
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
            except Exception as exc:
                log.warning("poll_error", component="prime", error=str(exc))
                continue

            record = next(
                (r for r in versions if r.get("generation") == offspring_gen), None
            )
            if record is None:
                continue

            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                break

        log.info(
            "generation_result",
            component="prime",
            generation=offspring_gen,
            outcome=outcome,
        )

        # Step 9: Decide
        viability = record.get("viability", {})
        viability_status = viability.get("status", "non-viable")

        if viability_status == "viable":
            try:
                await supervisor.promote(offspring_gen)
                log.info("generation_promoted", component="prime", generation=offspring_gen)
            except Exception as exc:
                log.error("promote_failed", component="prime", error=str(exc))
            # Successful promotion — stop
            return
        else:
            try:
                await supervisor.rollback(offspring_gen)
                log.info("generation_rolled_back", component="prime", generation=offspring_gen)
            except Exception as exc:
                log.error("rollback_failed", component="prime", error=str(exc))

            retry_count += 1
            gen_count += 1

            if retry_count > max_retries:
                log.warning("max_retries_exhausted", component="prime")
                return

            # Prepare failure context for informed retry
            diagnostics = viability.get("diagnostics", {})
            failed_files: dict[str, str] = {}
            for rel_path in file_list:
                if rel_path == "manifest.json":
                    continue
                file_path = artifact_dir / rel_path
                if file_path.is_file():
                    try:
                        failed_files[rel_path] = file_path.read_text(encoding="utf-8")
                    except Exception:
                        pass

            failure_context = {
                "failed_generation": offspring_gen,
                "diagnostics": diagnostics,
                "failed_files": failed_files,
            }

    log.warning("generation_loop_complete", component="prime")
