"""Generation loop: orchestrates LLM calls, file writing, and Supervisor interaction."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

CAMBRIAN_SPEC_PATH = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
CAMBRIAN_MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
CAMBRIAN_MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
CAMBRIAN_MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
WORKSPACE = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
ARTIFACTS_ROOT = Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace"))


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
    from src.generate import (
        ParseError,
        build_fresh_prompt,
        build_informed_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        parse_files,
        select_model,
    )
    from src.manifest import build_manifest, compute_artifact_hash, compute_spec_hash, write_manifest
    from src.supervisor import SupervisorClient

    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
    supervisor = SupervisorClient(supervisor_url)

    spec_path = Path(CAMBRIAN_SPEC_PATH)
    spec_content = spec_path.read_text()
    spec_hash = compute_spec_hash(spec_path)

    log.info(
        "generation_loop_starting",
        component="prime",
        spec_path=str(spec_path),
        spec_hash=spec_hash,
    )

    total_gens = 0
    consecutive_failures = 0

    while total_gens < CAMBRIAN_MAX_GENS and consecutive_failures < CAMBRIAN_MAX_RETRIES:
        # Step 1: Determine generation number
        history = await supervisor.get_versions()
        if history:
            next_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            next_gen = 1

        if history:
            parent_gen = max(r.get("generation", 0) for r in history)
        else:
            parent_gen = 0

        prime_module._prime_status = "generating"
        log.info(
            "generating",
            component="prime",
            generation=next_gen,
            parent=parent_gen,
            retry_count=consecutive_failures,
        )

        # Step 3+4: Build prompt and call LLM
        retry_count = consecutive_failures
        model = select_model(retry_count)

        # Gather failed context if this is a retry
        failed_files: dict[str, str] = {}
        diagnostics: dict[str, Any] | None = None
        failed_gen_num: int | None = None

        if retry_count > 0 and history:
            # Find most recent failed generation
            failed_records = [r for r in history if r.get("outcome") in ("failed",)]
            if failed_records:
                last_failed = max(failed_records, key=lambda r: r.get("generation", 0))
                failed_gen_num = last_failed.get("generation")
                viability = last_failed.get("viability", {})
                if viability:
                    diagnostics = viability.get("diagnostics")

                # Read failed source files from disk
                if failed_gen_num is not None:
                    failed_dir = ARTIFACTS_ROOT / f"gen-{failed_gen_num}"
                    if failed_dir.exists():
                        for fpath in failed_dir.rglob("*"):
                            if fpath.is_file():
                                rel = str(fpath.relative_to(failed_dir))
                                try:
                                    failed_files[rel] = fpath.read_text()
                                except Exception:
                                    pass

        if retry_count > 0 and diagnostics:
            prompt = build_informed_retry_prompt(
                spec_content=spec_content,
                history=history,
                generation=next_gen,
                parent=parent_gen,
                failed_gen=failed_gen_num or (next_gen - 1),
                diagnostics=diagnostics,
                failed_files=failed_files,
            )
        else:
            prompt = build_fresh_prompt(
                spec_content=spec_content,
                history=history,
                generation=next_gen,
                parent=parent_gen,
            )

        # Parse with repair loop
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_error_count = 0

        raw_response = await call_llm(prompt, model)
        token_usage = {"input": 0, "output": 0}  # Will be updated

        for parse_attempt in range(CAMBRIAN_MAX_PARSE_RETRIES + 1):
            try:
                parsed_files = parse_files(raw_response)
                break
            except ParseError as pe:
                parse_error_count += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    generation=next_gen,
                    attempt=parse_attempt,
                    error=str(pe),
                )
                if parse_attempt < CAMBRIAN_MAX_PARSE_RETRIES:
                    repair_prompt = build_parse_repair_prompt(str(pe), raw_response)
                    raw_response = await call_llm(repair_prompt, model)
                else:
                    # All parse repairs exhausted — count as generation failure
                    log.error(
                        "parse_repair_exhausted",
                        component="prime",
                        generation=next_gen,
                    )
                    parsed_files = None
                    break

        if parsed_files is None:
            consecutive_failures += 1
            total_gens += 1
            prime_module._prime_status = "idle"
            continue

        # Step 6: Write files to workspace
        artifact_dir = ARTIFACTS_ROOT / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        shutil.copy2(spec_path, spec_dest)

        # Build file list
        files_list: list[str] = []
        for fpath in artifact_dir.rglob("*"):
            if fpath.is_file():
                rel = str(fpath.relative_to(artifact_dir))
                files_list.append(rel)

        # Ensure manifest.json is in the list (will be written next)
        if "manifest.json" not in files_list:
            files_list.append("manifest.json")

        artifact_hash = compute_artifact_hash(artifact_dir, files_list)

        # Extract contracts from spec
        from src.manifest import extract_contracts_from_spec
        contracts = extract_contracts_from_spec(spec_content)

        manifest_data = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_hash=artifact_hash,
            producer_model=model,
            token_usage=token_usage,
            files=files_list,
            spec_file_path=f"spec/{spec_path.name}",
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest_data)

        # Step 7: POST /spawn
        prime_module._prime_status = "verifying"
        log.info(
            "spawning",
            component="prime",
            generation=next_gen,
            artifact_path=f"gen-{next_gen}",
        )

        spawn_result = await supervisor.spawn(
            spec_hash=spec_hash,
            generation=next_gen,
            artifact_path=f"gen-{next_gen}",
        )

        if not spawn_result.get("ok"):
            log.error(
                "spawn_failed",
                component="prime",
                generation=next_gen,
                error=spawn_result.get("error"),
            )
            consecutive_failures += 1
            total_gens += 1
            prime_module._prime_status = "idle"
            continue

        # Step 8: Poll for outcome
        log.info("polling_for_outcome", component="prime", generation=next_gen)
        outcome_record: dict[str, Any] | None = None

        while True:
            await asyncio.sleep(2)
            versions = await supervisor.get_versions()
            record = next(
                (r for r in versions if r.get("generation") == next_gen),
                None,
            )
            if record and record.get("outcome") != "in_progress":
                outcome_record = record
                break

        # Step 9: Decide
        viability_report = outcome_record.get("viability", {}) if outcome_record else {}
        status = viability_report.get("status", "non-viable")

        if status == "viable":
            log.info("promoting", component="prime", generation=next_gen)
            await supervisor.promote(next_gen)
            consecutive_failures = 0
            total_gens += 1
            # Done — one successful promotion per loop
            break
        else:
            log.info(
                "rolling_back",
                component="prime",
                generation=next_gen,
                failure_stage=viability_report.get("failure_stage"),
            )
            await supervisor.rollback(next_gen)
            consecutive_failures += 1
            total_gens += 1

    prime_module._prime_status = "idle"
    log.info(
        "generation_loop_done",
        component="prime",
        total_gens=total_gens,
        consecutive_failures=consecutive_failures,
    )
