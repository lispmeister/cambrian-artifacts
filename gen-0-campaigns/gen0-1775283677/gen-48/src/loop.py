"""Generation loop — orchestrates the full generate/verify/decide cycle."""

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
        ParseError,
        build_fresh_prompt,
        build_informed_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        get_model,
        parse_files,
    )
    from src.manifest import build_manifest, compute_spec_hash, write_manifest

    generation_number = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )
    spec_path = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
    workspace = Path("/workspace")
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    model_name = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    escalation_model = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error(
            "anthropic_api_key_missing_fatal",
            component="prime",
            generation=generation_number,
        )
        return

    supervisor = SupervisorClient(supervisor_url)

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)

    retry_count = 0
    gen_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        # Step 1: Determine generation number
        try:
            history = await supervisor.get_versions()
        except Exception as exc:
            log.error(
                "supervisor_unreachable",
                component="prime",
                generation=generation_number,
                error=str(exc),
            )
            await asyncio.sleep(2)
            continue

        if history:
            offspring_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            offspring_gen = 1

        parent_gen = offspring_gen - 1

        log.info(
            "generation_starting",
            component="prime",
            generation=generation_number,
            offspring_generation=offspring_gen,
            retry_count=retry_count,
        )

        # Step 3 & 4: Build prompt and call LLM
        if retry_count == 0 or failed_artifact_path is None:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=history,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
            )
        else:
            failed_files = _read_failed_files(failed_artifact_path)
            system_msg, user_msg = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=history,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
                failed_files=failed_files,
                diagnostics=failed_diagnostics or {},
            )

        current_model = get_model(retry_count)

        # Parse with repair loop
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        try:
            raw_response = await call_llm(
                system_msg=system_msg,
                user_msg=user_msg,
                model=current_model,
            )
        except Exception as exc:
            log.error(
                "llm_call_failed",
                component="prime",
                generation=generation_number,
                error=str(exc),
            )
            retry_count += 1
            gen_count += 1
            continue

        # Try parsing
        parse_ok = False
        while not parse_ok and parse_repair_count <= max_parse_retries:
            try:
                parsed_files = parse_files(raw_response)
                parse_ok = True
            except ParseError as pe:
                if parse_repair_count >= max_parse_retries:
                    log.error(
                        "parse_repair_exhausted",
                        component="prime",
                        generation=generation_number,
                        error=str(pe),
                    )
                    break
                # Attempt repair
                log.warning(
                    "parse_error_attempting_repair",
                    component="prime",
                    generation=generation_number,
                    parse_repair_count=parse_repair_count,
                    error=str(pe),
                )
                repair_system, repair_user = build_parse_repair_prompt(
                    parse_error_message=str(pe),
                    raw_response=raw_response,
                )
                try:
                    raw_response = await call_llm(
                        system_msg=repair_system,
                        user_msg=repair_user,
                        model=current_model,
                    )
                except Exception as exc:
                    log.error(
                        "parse_repair_llm_failed",
                        component="prime",
                        generation=generation_number,
                        error=str(exc),
                    )
                    break
                parse_repair_count += 1

        if not parse_ok or parsed_files is None:
            retry_count += 1
            gen_count += 1
            continue

        # Step 6: Write files to workspace
        artifact_dir = workspace / f"gen-{offspring_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            file_path = artifact_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest = artifact_dir / "spec" / spec_path.name
        spec_dest.parent.mkdir(parents=True, exist_ok=True)
        spec_dest.write_bytes(spec_path.read_bytes())

        # Get token usage from last LLM call (stored globally)
        from src import generate as gen_module
        token_usage = gen_module._last_token_usage or {"input": 0, "output": 0}

        # Build file list
        files_list = []
        for f in artifact_dir.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(artifact_dir))
                files_list.append(rel)

        # Write manifest
        manifest_data = build_manifest(
            artifact_root=artifact_dir,
            files=files_list,
            generation=offspring_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            spec_content=spec_content,
            token_usage=token_usage,
            model=current_model,
        )
        write_manifest(artifact_dir, manifest_data)

        # Step 7: POST /spawn
        artifact_path = f"gen-{offspring_gen}"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=offspring_gen,
                artifact_path=artifact_path,
            )
            log.info(
                "spawn_requested",
                component="prime",
                generation=generation_number,
                offspring_generation=offspring_gen,
                container_id=spawn_result.get("container-id", ""),
            )
        except Exception as exc:
            log.error(
                "spawn_failed",
                component="prime",
                generation=generation_number,
                error=str(exc),
            )
            retry_count += 1
            gen_count += 1
            continue

        # Step 8: Poll until outcome != in_progress
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
            except Exception as exc:
                log.warning(
                    "poll_supervisor_error",
                    component="prime",
                    generation=generation_number,
                    error=str(exc),
                )
                continue

            record = None
            for r in versions:
                if r.get("generation") == offspring_gen:
                    record = r
                    break

            if record is None:
                continue

            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                break

        # Step 9: Decide
        viability = record.get("viability", {})
        status = viability.get("status", "non-viable")

        if status == "viable":
            try:
                await supervisor.promote(offspring_gen)
                log.info(
                    "generation_promoted",
                    component="prime",
                    generation=generation_number,
                    offspring_generation=offspring_gen,
                )
            except Exception as exc:
                log.error(
                    "promote_failed",
                    component="prime",
                    generation=generation_number,
                    error=str(exc),
                )
            return  # Done
        else:
            try:
                await supervisor.rollback(offspring_gen)
                log.info(
                    "generation_rolled_back",
                    component="prime",
                    generation=generation_number,
                    offspring_generation=offspring_gen,
                )
            except Exception as exc:
                log.error(
                    "rollback_failed",
                    component="prime",
                    generation=generation_number,
                    error=str(exc),
                )

            failed_artifact_path = artifact_dir
            failed_diagnostics = viability.get("diagnostics")
            retry_count += 1
            gen_count += 1

    log.info(
        "generation_loop_exhausted",
        component="prime",
        generation=generation_number,
        gen_count=gen_count,
        retry_count=retry_count,
    )


def _read_failed_files(artifact_path: Path) -> dict[str, str]:
    """Read source files from a failed artifact directory."""
    result: dict[str, str] = {}
    if not artifact_path.exists():
        return result
    for f in artifact_path.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".txt", ".md", ".json", ".toml", ".cfg", ".ini"):
            rel = str(f.relative_to(artifact_path))
            if rel == "manifest.json":
                continue
            try:
                result[rel] = f.read_text(encoding="utf-8")
            except Exception:
                pass
    return result
