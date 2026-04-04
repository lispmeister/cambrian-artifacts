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

WORKSPACE = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
SPEC_PATH = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
SUPERVISOR_URL = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))


async def generation_loop() -> None:
    """Main generation loop."""
    from src.supervisor import SupervisorClient
    from src.generate import (
        build_fresh_prompt,
        build_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        parse_files,
        ParseError,
        get_model,
    )
    from src.manifest import build_manifest, write_manifest, compute_spec_hash

    supervisor = SupervisorClient(SUPERVISOR_URL)

    gen_count = 0
    retry_count = 0

    spec_content = SPEC_PATH.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(SPEC_PATH)

    log.info("loop_starting", component="prime", spec_hash=spec_hash)

    while gen_count < MAX_GENS and retry_count <= MAX_RETRIES:
        # Step 1: determine generation number
        history = await supervisor.get_versions()
        if history:
            next_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        log.info(
            "loop_generation_starting",
            component="prime",
            generation=next_gen,
            retry_count=retry_count,
        )

        # Step 3/4: build prompt and call LLM
        model = get_model(retry_count)

        if retry_count == 0:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                history=history,
                generation=next_gen,
                parent=parent_gen,
            )
        else:
            # Load failed code from previous workspace
            failed_gen = next_gen - 1
            failed_dir = WORKSPACE / f"gen-{failed_gen}"
            failed_files: dict[str, str] = {}
            if failed_dir.exists():
                for fp in failed_dir.rglob("*"):
                    if fp.is_file():
                        rel = str(fp.relative_to(failed_dir))
                        try:
                            failed_files[rel] = fp.read_text(encoding="utf-8")
                        except Exception:
                            pass

            # Get diagnostics from last record
            diagnostics: dict[str, Any] = {}
            if history:
                last = max(history, key=lambda r: r.get("generation", 0))
                viability = last.get("viability", {})
                diagnostics = viability.get("diagnostics", {})

            system_msg, user_msg = build_retry_prompt(
                spec_content=spec_content,
                history=history,
                generation=next_gen,
                parent=parent_gen,
                failed_gen=failed_gen,
                failed_files=failed_files,
                diagnostics=diagnostics,
            )

        # Parse with repair loop
        raw_response = None
        files: dict[str, str] = {}
        parse_ok = False
        parse_repair_count = 0

        for attempt in range(MAX_PARSE_RETRIES + 1):
            try:
                raw_response, token_usage = await call_llm(
                    system_msg=system_msg,
                    user_msg=user_msg,
                    model=model,
                )
                files = parse_files(raw_response)
                parse_ok = True
                break
            except ParseError as e:
                parse_repair_count += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    generation=next_gen,
                    attempt=attempt,
                    error=str(e),
                )
                if parse_repair_count > MAX_PARSE_RETRIES:
                    break
                if raw_response is not None:
                    repair_msg = build_parse_repair_prompt(
                        parse_error=str(e),
                        raw_response=raw_response,
                    )
                    system_msg, user_msg = system_msg, repair_msg
                else:
                    break

        if not parse_ok:
            log.error("parse_failed_all_retries", component="prime", generation=next_gen)
            retry_count += 1
            gen_count += 1
            continue

        # Step 6: write files to workspace
        artifact_dir = WORKSPACE / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / SPEC_PATH.name
        shutil.copy2(SPEC_PATH, spec_dest)

        # Compute file list
        file_list = []
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                file_list.append(str(fp.relative_to(artifact_dir)))

        # Build and write manifest
        manifest = build_manifest(
            artifact_root=artifact_dir,
            files=file_list,
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            producer_model=model,
            token_usage=token_usage,
            spec_content=spec_content,
        )
        write_manifest(artifact_dir, manifest)

        # Re-compute file list to include manifest.json
        file_list = []
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                file_list.append(str(fp.relative_to(artifact_dir)))

        log.info(
            "artifact_written",
            component="prime",
            generation=next_gen,
            artifact_dir=str(artifact_dir),
        )

        # Step 7: POST /spawn
        spawn_result = await supervisor.spawn(
            spec_hash=spec_hash,
            generation=next_gen,
            artifact_path=f"gen-{next_gen}",
        )
        log.info("spawn_result", component="prime", result=spawn_result)

        # Step 8: poll until outcome != in_progress
        log.info("polling_for_outcome", component="prime", generation=next_gen)
        record: dict[str, Any] = {}
        while True:
            versions = await supervisor.get_versions()
            for r in versions:
                if r.get("generation") == next_gen:
                    record = r
                    break
            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                break
            await asyncio.sleep(2)

        viability = record.get("viability", {})
        status = viability.get("status", "non-viable")

        log.info(
            "generation_outcome",
            component="prime",
            generation=next_gen,
            outcome=outcome,
            viability_status=status,
        )

        gen_count += 1

        if status == "viable":
            # Promote
            await supervisor.promote(next_gen)
            log.info("promoted", component="prime", generation=next_gen)
            retry_count = 0
            # One successful promotion per loop run
            break
        else:
            # Rollback
            await supervisor.rollback(next_gen)
            log.info("rolled_back", component="prime", generation=next_gen)
            retry_count += 1

    log.info(
        "loop_done",
        component="prime",
        gen_count=gen_count,
        retry_count=retry_count,
    )
