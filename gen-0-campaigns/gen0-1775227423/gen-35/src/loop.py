"""Generation loop implementation."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

import anthropic
import structlog

from src.generate import (
    ParseError,
    build_fresh_prompt,
    build_parse_repair_prompt,
    build_retry_prompt,
    compute_next_generation,
    get_max_gens,
    get_max_parse_retries,
    get_max_retries,
    get_system_prompt,
    parse_files,
    select_model,
)
from src.manifest import (
    build_manifest,
    compute_artifact_hash,
    compute_spec_hash,
    extract_contracts_from_spec,
    write_manifest,
)
from src.supervisor import SupervisorClient

logger = structlog.get_logger()


async def call_llm(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    user_message: str,
) -> tuple[str, dict[str, int]]:
    """Call the LLM using streaming and return response text + token usage."""
    async with client.messages.stream(
        model=model,
        max_tokens=16384,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text += block.text

    usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }
    return text, usage


async def generation_loop(
    supervisor: SupervisorClient,
    status_holder: dict[str, str],
) -> None:
    """Run the generation loop."""
    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    spec_path = Path(spec_path_str)
    workspace = Path("/workspace")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("missing_api_key", component="prime")
        return

    llm_client = anthropic.AsyncAnthropic(api_key=api_key)

    max_retries = get_max_retries()
    max_gens = get_max_gens()
    max_parse_retries = get_max_parse_retries()

    consecutive_failures = 0
    total_gens = 0
    current_spec_hash: str | None = None

    while total_gens < max_gens and consecutive_failures < max_retries:
        try:
            status_holder["status"] = "generating"

            # Step 1: Determine generation number
            records = await supervisor.get_versions()
            gen_number = compute_next_generation(records)

            # Find parent
            parent_gen = 0
            if records:
                promoted = [r for r in records if r.get("outcome") == "promoted"]
                if promoted:
                    parent_gen = max(r.get("generation", 0) for r in promoted)
                else:
                    parent_gen = max(r.get("generation", 0) for r in records)

            # Step 2: Read spec
            if not spec_path.exists():
                logger.error("spec_not_found", component="prime", path=str(spec_path))
                return

            spec_content = spec_path.read_text()
            spec_hash = compute_spec_hash(spec_path)

            # Reset consecutive failures if spec changed
            if current_spec_hash is not None and current_spec_hash != spec_hash:
                consecutive_failures = 0
            current_spec_hash = spec_hash

            records_json = json.dumps(records, indent=2)

            # Step 3: Build prompt
            retry_count = consecutive_failures
            model = select_model(retry_count)

            # Check if this is an informed retry
            failed_files: dict[str, str] = {}
            diagnostics: dict[str, Any] | None = None

            if retry_count > 0 and records:
                last_record = max(records, key=lambda r: r.get("generation", 0))
                viability = last_record.get("viability", {})
                if viability and viability.get("diagnostics"):
                    diagnostics = viability["diagnostics"]
                    # Read failed source code from disk
                    prev_gen = last_record.get("generation", 0)
                    prev_artifact_dir = workspace / f"gen-{prev_gen}"
                    if prev_artifact_dir.exists():
                        for fpath in prev_artifact_dir.rglob("*"):
                            if fpath.is_file() and fpath.name != "manifest.json":
                                rel = str(fpath.relative_to(prev_artifact_dir))
                                try:
                                    failed_files[rel] = fpath.read_text()
                                except Exception:
                                    pass

            if diagnostics and retry_count > 0:
                user_message = build_retry_prompt(
                    spec_content=spec_content,
                    generation_records_json=records_json,
                    generation_number=gen_number,
                    parent_generation=parent_gen,
                    prev_generation=gen_number - 1,
                    diagnostics=diagnostics,
                    failed_files=failed_files,
                )
            else:
                user_message = build_fresh_prompt(
                    spec_content=spec_content,
                    generation_records_json=records_json,
                    generation_number=gen_number,
                    parent_generation=parent_gen,
                )

            system_prompt = get_system_prompt()

            logger.info(
                "calling_llm",
                component="prime",
                generation=gen_number,
                model=model,
                retry_count=retry_count,
            )

            # Step 4: Call LLM
            raw_response, token_usage = await call_llm(
                llm_client, model, system_prompt, user_message
            )

            # Step 5: Parse response with repair loop
            parsed_files: dict[str, str] | None = None
            parse_attempts = 0

            while parse_attempts <= max_parse_retries:
                try:
                    parsed_files = parse_files(raw_response)
                    break
                except ParseError as pe:
                    parse_attempts += 1
                    if parse_attempts > max_parse_retries:
                        logger.error(
                            "parse_repair_exhausted",
                            component="prime",
                            generation=gen_number,
                            error=str(pe),
                        )
                        break
                    logger.warning(
                        "parse_repair_attempt",
                        component="prime",
                        generation=gen_number,
                        attempt=parse_attempts,
                        error=str(pe),
                    )
                    repair_prompt = build_parse_repair_prompt(str(pe), raw_response)
                    raw_response, repair_usage = await call_llm(
                        llm_client, model, system_prompt, repair_prompt
                    )
                    token_usage["input"] += repair_usage["input"]
                    token_usage["output"] += repair_usage["output"]

            if parsed_files is None:
                total_gens += 1
                consecutive_failures += 1
                logger.error(
                    "generation_failed_parse",
                    component="prime",
                    generation=gen_number,
                )
                continue

            # Step 6: Write files to workspace
            artifact_dir = workspace / f"gen-{gen_number}"
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            artifact_dir.mkdir(parents=True, exist_ok=True)

            for file_path, file_content in parsed_files.items():
                full_path = artifact_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(file_content)

            # Copy spec file
            spec_dest_dir = artifact_dir / spec_path.parent.name
            spec_dest_dir.mkdir(parents=True, exist_ok=True)
            spec_dest = spec_dest_dir / spec_path.name
            shutil.copy2(spec_path, spec_dest)

            # Collect all file paths
            all_files: list[str] = []
            for fpath in artifact_dir.rglob("*"):
                if fpath.is_file():
                    rel = str(fpath.relative_to(artifact_dir))
                    all_files.append(rel)

            if "manifest.json" not in all_files:
                all_files.append("manifest.json")

            # Extract contracts from spec
            contracts = extract_contracts_from_spec(spec_content)

            # Compute artifact hash (before writing manifest)
            artifact_hash = compute_artifact_hash(artifact_dir, all_files)

            # Build and write manifest
            manifest = build_manifest(
                generation=gen_number,
                parent_generation=parent_gen,
                spec_hash=spec_hash,
                artifact_hash=artifact_hash,
                producer_model=model,
                token_usage=token_usage,
                files=sorted(all_files),
                contracts=contracts,
            )
            write_manifest(artifact_dir, manifest)

            logger.info(
                "artifact_written",
                component="prime",
                generation=gen_number,
                files=len(all_files),
            )

            # Step 7: Request verification
            status_holder["status"] = "verifying"
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=gen_number,
                artifact_path=f"gen-{gen_number}",
            )

            if not spawn_result.get("ok", False):
                logger.error(
                    "spawn_failed",
                    component="prime",
                    generation=gen_number,
                    error=spawn_result.get("error", "unknown"),
                )
                total_gens += 1
                consecutive_failures += 1
                continue

            # Step 8: Poll until outcome != in_progress
            while True:
                await asyncio.sleep(2)
                versions = await supervisor.get_versions()
                gen_record = None
                for r in versions:
                    if r.get("generation") == gen_number:
                        gen_record = r
                        break

                if gen_record is None:
                    continue

                outcome = gen_record.get("outcome", "in_progress")
                if outcome != "in_progress":
                    break

            # Step 9: Decide
            viability = gen_record.get("viability", {}) if gen_record else {}
            viability_status = viability.get("status", "non-viable") if viability else "non-viable"

            total_gens += 1

            if viability_status == "viable":
                await supervisor.promote(gen_number)
                consecutive_failures = 0
                logger.info(
                    "generation_promoted",
                    component="prime",
                    generation=gen_number,
                )
                status_holder["status"] = "idle"
                return
            else:
                await supervisor.rollback(gen_number)
                consecutive_failures += 1
                logger.warning(
                    "generation_failed",
                    component="prime",
                    generation=gen_number,
                    failure_stage=viability.get("failure_stage", "unknown") if viability else "unknown",
                )

        except Exception as exc:
            logger.error(
                "generation_loop_error",
                component="prime",
                error=str(exc),
            )
            total_gens += 1
            consecutive_failures += 1
            await asyncio.sleep(2)

    status_holder["status"] = "idle"
    logger.info(
        "generation_loop_stopped",
        component="prime",
        total_gens=total_gens,
        consecutive_failures=consecutive_failures,
    )
