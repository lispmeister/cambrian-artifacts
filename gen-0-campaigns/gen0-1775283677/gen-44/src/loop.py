"""Generation loop — orchestrates LLM calls, file writing, and Supervisor communication."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
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
        get_model,
        parse_files,
    )
    from src.manifest import build_manifest, write_manifest, compute_spec_hash

    # Configuration
    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    workspace = os.environ.get("CAMBRIAN_WORKSPACE", "/workspace")
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))

    spec_path = Path(spec_path_str)
    if not spec_path.exists():
        log.error("spec_not_found", path=str(spec_path), component="prime")
        return

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)

    supervisor = SupervisorClient(supervisor_url)

    retry_count = 0
    gen_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        # Step 1: Determine generation number
        history = await supervisor.get_versions()
        if history:
            next_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        log.info("generation_starting", component="prime", generation=next_gen,
                 retry_count=retry_count)

        # Step 3: Build prompt
        if retry_count == 0 or failed_artifact_path is None:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=history,
                generation=next_gen,
                parent=parent_gen,
            )
        else:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path.exists():
                for fpath in sorted(failed_artifact_path.rglob("*")):
                    if fpath.is_file() and fpath.name != "manifest.json":
                        rel = str(fpath.relative_to(failed_artifact_path))
                        try:
                            failed_files[rel] = fpath.read_text(encoding="utf-8")
                        except Exception:
                            pass

            system_msg, user_msg = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=history,
                generation=next_gen,
                parent=parent_gen,
                failed_generation=next_gen - 1,
                diagnostics=failed_diagnostics or {},
                failed_files=failed_files,
            )

        # Step 4: Call LLM (with parse repair loop)
        model = get_model(retry_count)
        log.info("calling_llm", component="prime", model=model, generation=next_gen)

        raw_response = await call_llm(system_msg, user_msg, model)

        # Step 5: Parse response (with repair loop)
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        while parsed_files is None:
            try:
                parsed_files = parse_files(raw_response)
            except ParseError as pe:
                log.warning("parse_error", component="prime", error=str(pe),
                            parse_repair_count=parse_repair_count)
                if parse_repair_count >= max_parse_retries:
                    log.error("parse_repair_exhausted", component="prime")
                    # Count as generation failure
                    retry_count += 1
                    gen_count += 1
                    failed_artifact_path = None
                    failed_diagnostics = None
                    break
                # Attempt parse repair
                repair_system, repair_user = build_parse_repair_prompt(
                    parse_error_message=str(pe),
                    raw_response=raw_response,
                )
                raw_response = await call_llm(repair_system, repair_user, model)
                parse_repair_count += 1

        if parsed_files is None:
            continue

        # Step 6: Write files to workspace
        artifact_dir = Path(workspace) / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Write parsed files
        for rel_path, content in parsed_files.items():
            target = artifact_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        shutil.copy2(spec_path, spec_dest)

        # Collect all files
        all_files: list[str] = []
        for fpath in sorted(artifact_dir.rglob("*")):
            if fpath.is_file():
                all_files.append(str(fpath.relative_to(artifact_dir)))

        # Extract contracts from spec
        from src.manifest import extract_contracts_from_spec
        contracts = extract_contracts_from_spec(spec_content)

        # Get token usage from LLM call (stored as module-level var)
        from src import generate as gen_module
        token_usage = getattr(gen_module, "_last_token_usage", {"input": 0, "output": 0})

        model_name = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
        if retry_count >= 1:
            model_name = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

        manifest = build_manifest(
            artifact_root=artifact_dir,
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            files=all_files,
            producer_model=model_name,
            token_usage=token_usage,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest)

        log.info("artifact_written", component="prime", generation=next_gen,
                 path=str(artifact_dir))

        # Step 7: POST /spawn
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn_requested", component="prime", generation=next_gen,
                     result=spawn_result)
        except Exception as exc:
            log.error("spawn_failed", component="prime", error=str(exc))
            retry_count += 1
            gen_count += 1
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("polling_for_result", component="prime", generation=next_gen)
        viability_report: dict[str, Any] | None = None
        while True:
            await asyncio.sleep(2)
            versions = await supervisor.get_versions()
            record = next(
                (r for r in versions if r.get("generation") == next_gen), None
            )
            if record is None:
                continue
            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                viability_report = record.get("viability")
                break

        # Step 9: Decide
        if viability_report is None:
            log.error("no_viability_report", component="prime", generation=next_gen)
            await supervisor.rollback(next_gen)
            retry_count += 1
            gen_count += 1
            failed_artifact_path = artifact_dir
            failed_diagnostics = {}
            continue

        status = viability_report.get("status", "non-viable")
        if status == "viable":
            log.info("generation_viable", component="prime", generation=next_gen)
            await supervisor.promote(next_gen)
            log.info("generation_promoted", component="prime", generation=next_gen)
            retry_count = 0  # Reset on success
            gen_count += 1
            # Done after one successful promotion per loop
            break
        else:
            diagnostics = viability_report.get("diagnostics", {})
            log.warning("generation_non_viable", component="prime", generation=next_gen,
                        stage=diagnostics.get("stage", "unknown"))
            await supervisor.rollback(next_gen)
            failed_artifact_path = artifact_dir
            failed_diagnostics = diagnostics
            retry_count += 1
            gen_count += 1

    log.info("generation_loop_complete", component="prime", gen_count=gen_count,
             retry_count=retry_count)
