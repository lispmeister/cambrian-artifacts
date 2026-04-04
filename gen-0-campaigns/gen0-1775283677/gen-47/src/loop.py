"""Generation loop — orchestrates the LLM generation cycle."""
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
    from src.manifest import build_manifest, write_manifest, compute_spec_hash, extract_contracts_from_spec

    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    spec_path = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
    workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")

    supervisor = SupervisorClient(supervisor_url)

    # Read spec
    if not spec_path.exists():
        log.error("spec_not_found", component="prime", path=str(spec_path))
        return

    spec_content = spec_path.read_text()
    spec_hash = compute_spec_hash(spec_content)

    log.info("generation_loop_starting", component="prime", spec_hash=spec_hash)

    retry_count = 0
    gen_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        # Step 1: Determine generation number
        try:
            history = await supervisor.get_versions()
        except Exception as e:
            log.error("supervisor_unreachable", component="prime", error=str(e))
            await asyncio.sleep(5)
            continue

        if history:
            next_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1
        log.info("generation_starting", component="prime", generation=next_gen, retry_count=retry_count)

        # Step 3: Build prompt
        history_json = json.dumps(history, indent=2)
        if retry_count == 0 or failed_artifact_path is None:
            system_msg, user_msg = build_fresh_prompt(spec_content, history_json, next_gen, parent_gen)
        else:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path.exists():
                for fpath in failed_artifact_path.rglob("*"):
                    if fpath.is_file() and fpath.name != "manifest.json":
                        rel = str(fpath.relative_to(failed_artifact_path))
                        try:
                            failed_files[rel] = fpath.read_text()
                        except Exception:
                            pass
            system_msg, user_msg = build_retry_prompt(
                spec_content, history_json, next_gen, parent_gen,
                failed_files, failed_diagnostics or {}
            )

        # Step 4: Call LLM with parse repair loop
        model = get_model(retry_count)
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        try:
            raw_response = await call_llm(system_msg, user_msg, model)
        except Exception as e:
            log.error("llm_call_failed", component="prime", error=str(e))
            retry_count += 1
            gen_count += 1
            continue

        # Step 5: Parse response with repair loop
        while True:
            try:
                parsed_files = parse_files(raw_response)
                break
            except ParseError as pe:
                log.warning("parse_error", component="prime", error=str(pe), repair_attempt=parse_repair_count)
                if parse_repair_count >= max_parse_retries:
                    log.error("parse_repair_exhausted", component="prime")
                    break
                # Attempt repair
                repair_system, repair_user = build_parse_repair_prompt(str(pe), raw_response)
                try:
                    raw_response = await call_llm(repair_system, repair_user, model)
                    parse_repair_count += 1
                except Exception as e:
                    log.error("llm_repair_call_failed", component="prime", error=str(e))
                    break

        if parsed_files is None:
            log.error("all_parse_repairs_failed", component="prime")
            retry_count += 1
            gen_count += 1
            continue

        # Step 6: Write files to workspace
        artifact_dir = workspace / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            fpath = artifact_dir / rel_path
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(content)

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Collect all files
        all_files: list[str] = []
        for fpath in sorted(artifact_dir.rglob("*")):
            if fpath.is_file():
                rel = str(fpath.relative_to(artifact_dir))
                all_files.append(rel)

        if "manifest.json" not in all_files:
            all_files.append("manifest.json")

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Build manifest
        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_root=artifact_dir,
            files=all_files,
            model=model,
            token_input=0,
            token_output=0,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest)

        log.info("artifact_written", component="prime", generation=next_gen, path=str(artifact_dir))

        # Step 7: POST /spawn
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn_requested", component="prime", generation=next_gen, result=spawn_result)
        except Exception as e:
            log.error("spawn_failed", component="prime", error=str(e))
            retry_count += 1
            gen_count += 1
            continue

        # Step 8: Poll for outcome
        viability_report: dict[str, Any] | None = None
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
                record = next((r for r in versions if r.get("generation") == next_gen), None)
                if record and record.get("outcome") != "in_progress":
                    viability_report = record.get("viability")
                    break
            except Exception as e:
                log.warning("poll_error", component="prime", error=str(e))

        # Step 9: Decide
        gen_count += 1

        if viability_report and viability_report.get("status") == "viable":
            try:
                await supervisor.promote(next_gen)
                log.info("generation_promoted", component="prime", generation=next_gen)
            except Exception as e:
                log.error("promote_failed", component="prime", error=str(e))
            return  # Done after successful promotion
        else:
            log.warning("generation_non_viable", component="prime", generation=next_gen)
            try:
                await supervisor.rollback(next_gen)
            except Exception as e:
                log.error("rollback_failed", component="prime", error=str(e))

            failed_artifact_path = artifact_dir
            failed_diagnostics = (viability_report or {}).get("diagnostics")
            retry_count += 1

    log.info("generation_loop_finished", component="prime", gen_count=gen_count, retry_count=retry_count)
