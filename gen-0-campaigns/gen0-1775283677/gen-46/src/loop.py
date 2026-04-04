"""Generation loop — orchestrates LLM calls, file writing, and Supervisor interactions."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
    from src.generate import (
        ParseError,
        build_fresh_prompt,
        build_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        get_model,
        parse_files,
    )
    from src.manifest import build_manifest, write_manifest, compute_spec_hash
    from src.supervisor import SupervisorClient

    # Config
    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )
    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    workspace_root = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))

    spec_path = Path(spec_path_str)

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

    while gen_count < max_gens and retry_count <= max_retries:
        # Step 1: Determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as exc:
            log.error("supervisor_get_versions_failed", component="prime", error=str(exc))
            await asyncio.sleep(5)
            continue

        if versions:
            offspring_gen = max(v.get("generation", 0) for v in versions) + 1
            parent_gen = offspring_gen - 1
        else:
            offspring_gen = 1
            parent_gen = 0

        log.info(
            "generation_starting",
            component="prime",
            offspring_generation=offspring_gen,
            retry_count=retry_count,
        )

        prime_module._prime_status = "generating"

        # Step 3/4: Build prompt and call LLM with parse repair loop
        model = get_model(retry_count)

        failed_artifact_path: Path | None = None
        diagnostics: dict[str, Any] | None = None

        # Get the previous failed generation info if retrying
        if retry_count > 0 and versions:
            # Find the most recent failed generation
            failed_records = [
                v for v in versions
                if v.get("outcome") in ("failed",)
            ]
            if failed_records:
                last_failed = max(failed_records, key=lambda x: x.get("generation", 0))
                failed_gen_num = last_failed.get("generation", 0)
                failed_artifact_path = workspace_root / f"gen-{failed_gen_num}"
                viability = last_failed.get("viability", {})
                diagnostics = viability.get("diagnostics")

        # Build prompt
        if retry_count > 0 and failed_artifact_path and diagnostics:
            failed_files = _read_failed_files(failed_artifact_path)
            system_msg, user_msg = build_retry_prompt(
                spec_content=spec_content,
                generation_records=versions,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
                diagnostics=diagnostics,
                failed_files=failed_files,
            )
        else:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=versions,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
            )

        # LLM call with parse repair
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        for attempt in range(max_parse_retries + 1):
            try:
                if attempt == 0:
                    raw_response = await call_llm(system_msg, user_msg, model)
                else:
                    # Parse repair
                    assert raw_response is not None
                    repair_user_msg = build_parse_repair_prompt(
                        parse_error_message=str(last_parse_error),
                        raw_response=raw_response,
                    )
                    raw_response = await call_llm(system_msg, repair_user_msg, model)
                    parse_repair_count += 1

                parsed_files = parse_files(raw_response)
                break  # Success
            except ParseError as exc:
                last_parse_error = exc
                log.warning(
                    "parse_error",
                    component="prime",
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt >= max_parse_retries:
                    log.error("parse_repair_exhausted", component="prime")
                    break

        if parsed_files is None:
            log.error("parse_failed_after_repairs", component="prime")
            retry_count += 1
            gen_count += 1
            prime_module._prime_status = "idle"
            continue

        # Step 6: Write files to workspace
        artifact_dir = workspace_root / f"gen-{offspring_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for file_path, content in parsed_files.items():
            dest = artifact_dir / file_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_text(spec_content, encoding="utf-8")
        spec_rel_path = f"spec/{spec_path.name}"

        # Collect all files
        all_files: list[str] = []
        for f in artifact_dir.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(artifact_dir))
                all_files.append(rel)

        # Ensure manifest.json is in the list
        if "manifest.json" not in all_files:
            all_files.append("manifest.json")

        # Get token usage from LLM response (stored in call_llm)
        token_usage = {"input": 0, "output": 0}

        # Get model name
        producer_model = model

        # Extract contracts from spec
        from src.manifest import extract_contracts_from_spec
        contracts = extract_contracts_from_spec(spec_content)

        # Build manifest
        manifest = build_manifest(
            artifact_root=artifact_dir,
            generation=offspring_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            producer_model=producer_model,
            token_usage=token_usage,
            files=all_files,
            contracts=contracts,
        )

        write_manifest(artifact_dir, manifest)

        log.info(
            "artifact_written",
            component="prime",
            generation=offspring_gen,
            artifact_dir=str(artifact_dir),
        )

        # Step 7: POST /spawn
        prime_module._prime_status = "verifying"
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
            prime_module._prime_status = "idle"
            continue

        # Step 8: Poll until outcome != in_progress
        viability_status: str | None = None
        viability_report: dict[str, Any] | None = None

        for _ in range(300):  # Max ~10 minutes
            await asyncio.sleep(2)
            try:
                versions_poll = await supervisor.get_versions()
            except Exception as exc:
                log.warning("poll_error", component="prime", error=str(exc))
                continue

            # Find our generation record
            our_record = next(
                (v for v in versions_poll if v.get("generation") == offspring_gen),
                None,
            )
            if our_record is None:
                continue

            outcome = our_record.get("outcome", "in_progress")
            if outcome != "in_progress":
                viability = our_record.get("viability", {})
                viability_status = viability.get("status")
                viability_report = viability
                log.info(
                    "outcome_received",
                    component="prime",
                    generation=offspring_gen,
                    outcome=outcome,
                    viability_status=viability_status,
                )
                break

        # Step 9: Decide
        gen_count += 1

        if viability_status == "viable":
            try:
                promote_result = await supervisor.promote(generation=offspring_gen)
                log.info("promoted", component="prime", generation=offspring_gen, result=promote_result)
            except Exception as exc:
                log.error("promote_failed", component="prime", error=str(exc))
            retry_count = 0
            prime_module._prime_status = "idle"
            break  # Done for this Prime instance
        else:
            # Non-viable or unknown
            log.warning(
                "generation_non_viable",
                component="prime",
                generation=offspring_gen,
                viability_status=viability_status,
            )
            try:
                await supervisor.rollback(generation=offspring_gen)
            except Exception as exc:
                log.error("rollback_failed", component="prime", error=str(exc))

            retry_count += 1
            prime_module._prime_status = "idle"

    if gen_count >= max_gens:
        log.info("max_gens_reached", component="prime", gen_count=gen_count)
    if retry_count > max_retries:
        log.info("max_retries_reached", component="prime", retry_count=retry_count)

    import src.prime as pm
    pm._prime_status = "idle"


def _read_failed_files(artifact_path: Path) -> dict[str, str]:
    """Read source files from a failed artifact directory."""
    files: dict[str, str] = {}
    if not artifact_path.exists():
        return files
    for f in artifact_path.rglob("*"):
        if f.is_file() and f.name != "manifest.json":
            rel = str(f.relative_to(artifact_path))
            try:
                files[rel] = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return files
