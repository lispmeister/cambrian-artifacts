"""Generation loop — orchestrates LLM calls, file writing, and Supervisor interaction."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

CAMBRIAN_MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
CAMBRIAN_MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
CAMBRIAN_MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
CAMBRIAN_SPEC_PATH = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
CAMBRIAN_SUPERVISOR_URL = os.environ.get(
    "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
)
CAMBRIAN_ARTIFACTS_ROOT = os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace")


def get_next_generation(records: list[dict[str, Any]]) -> int:
    """Compute the next generation number from history."""
    if not records:
        return 1
    return max(r.get("generation", 0) for r in records) + 1


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
    from src.manifest import build_manifest, compute_spec_hash, write_manifest
    from src.supervisor import SupervisorClient

    spec_path = Path(CAMBRIAN_SPEC_PATH)
    if not spec_path.exists():
        log.error(
            "spec_not_found",
            component="prime",
            path=str(spec_path),
        )
        return

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)

    supervisor = SupervisorClient(CAMBRIAN_SUPERVISOR_URL)

    consecutive_failures = 0
    total_generations = 0
    retry_count = 0
    failed_context: dict[str, Any] | None = None

    while total_generations < CAMBRIAN_MAX_GENS and consecutive_failures < CAMBRIAN_MAX_RETRIES:
        # Step 1: Determine generation number
        records = await supervisor.get_versions()
        offspring_gen = get_next_generation(records)
        parent_gen = offspring_gen - 1

        log.info(
            "generation_starting",
            component="prime",
            offspring_generation=offspring_gen,
            retry_count=retry_count,
        )

        prime_module._prime_status = "generating"

        # Step 3: Build prompt
        if failed_context is not None:
            messages = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=records,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
                failed_context=failed_context,
            )
        else:
            messages = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=records,
                offspring_gen=offspring_gen,
                parent_gen=parent_gen,
            )

        # Step 4: Call LLM with parse repair loop
        model = get_model(retry_count)
        raw_response: str | None = None
        files: dict[str, str] | None = None
        parse_repair_count = 0

        while True:
            raw_response = await call_llm(model=model, messages=messages)
            try:
                files = parse_files(raw_response)
                break
            except ParseError as exc:
                log.warning(
                    "parse_error",
                    component="prime",
                    error=str(exc),
                    parse_repair_count=parse_repair_count,
                )
                parse_repair_count += 1
                if parse_repair_count >= CAMBRIAN_MAX_PARSE_RETRIES:
                    log.error(
                        "parse_repair_exhausted",
                        component="prime",
                        parse_repair_count=parse_repair_count,
                    )
                    # Count as a generation failure
                    consecutive_failures += 1
                    total_generations += 1
                    retry_count += 1
                    files = None
                    break
                # Try parse repair
                repair_messages = build_parse_repair_prompt(
                    parse_error_message=str(exc),
                    raw_response=raw_response,
                )
                messages = repair_messages

        if files is None:
            failed_context = {
                "diagnostics": {
                    "stage": "parse",
                    "summary": "Failed to parse LLM response after max parse retries",
                    "exit_code": None,
                    "failures": [],
                    "stdout_tail": "",
                    "stderr_tail": "",
                },
                "failed_files": {},
            }
            continue

        # Step 6: Write files to workspace
        artifact_root = Path(CAMBRIAN_ARTIFACTS_ROOT) / f"gen-{offspring_gen}"
        artifact_root.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            file_path = artifact_root / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_root / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Collect all files for manifest
        all_files: list[str] = []
        for f in artifact_root.rglob("*"):
            if f.is_file() and f.name != "manifest.json":
                rel = str(f.relative_to(artifact_root))
                all_files.append(rel)

        # Build and write manifest
        # (token usage will be filled in after we have it)
        token_usage = {"input": 0, "output": 0}
        manifest_data = build_manifest(
            artifact_root=artifact_root,
            files=all_files,
            generation=offspring_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            producer_model=model,
            token_usage=token_usage,
            spec_content=spec_content,
        )
        write_manifest(artifact_root, manifest_data)

        # Update files list to include manifest
        all_files_with_manifest = all_files + ["manifest.json"]

        prime_module._prime_status = "verifying"

        # Step 7: POST /spawn
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=offspring_gen,
                artifact_path=f"gen-{offspring_gen}",
            )
            log.info(
                "spawn_requested",
                component="prime",
                generation=offspring_gen,
                result=spawn_result,
            )
        except Exception as exc:
            log.error(
                "spawn_failed",
                component="prime",
                generation=offspring_gen,
                error=str(exc),
            )
            consecutive_failures += 1
            total_generations += 1
            retry_count += 1
            continue

        # Step 8: Poll for outcome
        outcome_record = await poll_for_outcome(supervisor, offspring_gen)

        viability = outcome_record.get("viability", {})
        status = viability.get("status", "non-viable")

        if status == "viable":
            # Step 9: Promote
            try:
                await supervisor.promote(offspring_gen)
                log.info(
                    "generation_promoted",
                    component="prime",
                    generation=offspring_gen,
                )
            except Exception as exc:
                log.error(
                    "promote_failed",
                    component="prime",
                    generation=offspring_gen,
                    error=str(exc),
                )
            consecutive_failures = 0
            retry_count = 0
            total_generations += 1
            prime_module._prime_status = "idle"
            return  # Done after one successful promotion
        else:
            # Rollback
            try:
                await supervisor.rollback(offspring_gen)
                log.info(
                    "generation_rolled_back",
                    component="prime",
                    generation=offspring_gen,
                )
            except Exception as exc:
                log.error(
                    "rollback_failed",
                    component="prime",
                    generation=offspring_gen,
                    error=str(exc),
                )

            consecutive_failures += 1
            total_generations += 1
            retry_count += 1

            # Prepare failure context for informed retry
            diagnostics = viability.get("diagnostics", {})
            failed_files: dict[str, str] = {}
            for rel_path_str, _content in files.items():
                file_path = artifact_root / rel_path_str
                if file_path.exists():
                    failed_files[rel_path_str] = file_path.read_text(encoding="utf-8")

            failed_context = {
                "diagnostics": diagnostics,
                "failed_files": failed_files,
                "failed_generation": offspring_gen,
            }

        prime_module._prime_status = "idle"

    log.info(
        "generation_loop_done",
        component="prime",
        total_generations=total_generations,
        consecutive_failures=consecutive_failures,
    )


async def poll_for_outcome(
    supervisor: Any, generation: int, poll_interval: float = 2.0
) -> dict[str, Any]:
    """Poll GET /versions until the generation's outcome is no longer in_progress."""
    while True:
        try:
            records = await supervisor.get_versions()
            for record in records:
                if record.get("generation") == generation:
                    outcome = record.get("outcome", "in_progress")
                    if outcome != "in_progress":
                        return record
        except Exception as exc:
            log.warning(
                "poll_error",
                component="prime",
                generation=generation,
                error=str(exc),
            )
        await asyncio.sleep(poll_interval)
