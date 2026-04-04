"""Generation loop — the main reproductive cycle of Prime."""
from __future__ import annotations

import asyncio
import json
import os
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
MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
PRIME_GENERATION = int(os.environ.get("CAMBRIAN_GENERATION", "0"))

POLL_INTERVAL = 2.0


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
    from src.supervisor import SupervisorClient
    from src.generate import build_fresh_prompt, build_retry_prompt, build_parse_repair_prompt, call_llm, parse_files, ParseError, SYSTEM_PROMPT
    from src.manifest import build_manifest, write_manifest, compute_spec_hash, extract_contracts_from_spec

    supervisor = SupervisorClient(SUPERVISOR_URL)
    consecutive_failures = 0
    total_generations = 0

    log.info("generation_loop_starting", component="prime", generation=PRIME_GENERATION)

    while total_generations < MAX_GENS and consecutive_failures < MAX_RETRIES:
        # Step 1: Determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as e:
            log.error("supervisor_unreachable", component="prime", error=str(e))
            await asyncio.sleep(5)
            continue

        if versions:
            next_gen = max(v.get("generation", 0) for v in versions) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        # Step 2: Read spec
        spec_content = SPEC_PATH.read_text()
        spec_hash = compute_spec_hash(SPEC_PATH)

        # Step 3 & 4: Build prompt and call LLM with parse repair loop
        retry_count = consecutive_failures
        model = ESCALATION_MODEL if retry_count >= 1 else MODEL

        # Read failed context if retrying
        failed_context: dict[str, Any] | None = None
        if retry_count > 0 and versions:
            # Find the most recent failed generation
            for v in sorted(versions, key=lambda x: x.get("generation", 0), reverse=True):
                if v.get("outcome") in ("failed",) and v.get("viability"):
                    viability = v["viability"]
                    if viability.get("status") == "non-viable" and "diagnostics" in viability:
                        failed_gen_num = v["generation"]
                        failed_path = ARTIFACTS_ROOT / f"gen-{failed_gen_num}"
                        failed_files: dict[str, str] = {}
                        if failed_path.exists():
                            for fp in failed_path.rglob("*"):
                                if fp.is_file() and fp.name != "manifest.json":
                                    try:
                                        rel = fp.relative_to(failed_path)
                                        failed_files[str(rel)] = fp.read_text(errors="replace")
                                    except Exception:
                                        pass
                        failed_context = {
                            "generation": failed_gen_num,
                            "diagnostics": viability["diagnostics"],
                            "files": failed_files,
                        }
                        break

        prime_module._status = "generating"

        # Build user message
        history_json = json.dumps(versions, indent=2)
        if failed_context is None or retry_count == 0:
            user_message = build_fresh_prompt(spec_content, history_json, next_gen, parent_gen)
        else:
            user_message = build_retry_prompt(
                spec_content, history_json, next_gen, parent_gen,
                failed_context["generation"], failed_context["diagnostics"],
                failed_context["files"]
            )

        # Parse repair loop
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        raw_response = await call_llm(SYSTEM_PROMPT, user_message, model)
        parse_error_msg = ""

        while parsed_files is None and parse_repair_count <= MAX_PARSE_RETRIES:
            try:
                parsed_files = parse_files(raw_response)
            except ParseError as e:
                parse_error_msg = str(e)
                log.warning("parse_error", component="prime", error=parse_error_msg,
                            attempt=parse_repair_count)
                if parse_repair_count >= MAX_PARSE_RETRIES:
                    break
                # Attempt repair
                repair_message = build_parse_repair_prompt(parse_error_msg, raw_response)
                raw_response = await call_llm(SYSTEM_PROMPT, repair_message, model)
                parse_repair_count += 1

        if parsed_files is None:
            log.error("parse_failed_all_repairs", component="prime", gen=next_gen)
            consecutive_failures += 1
            total_generations += 1
            prime_module._status = "idle"
            continue

        # Step 6: Write files to workspace
        prime_module._status = "verifying"
        artifact_dir = ARTIFACTS_ROOT / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / SPEC_PATH.name
        spec_dest.write_bytes(SPEC_PATH.read_bytes())

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Collect all files for manifest
        all_files: list[str] = []
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                rel = str(fp.relative_to(artifact_dir))
                all_files.append(rel)

        # Determine token usage from last LLM call (stored globally)
        from src.generate import _last_token_usage
        token_usage = _last_token_usage.copy()

        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_root=artifact_dir,
            files=all_files,
            model=model,
            token_usage=token_usage,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest)

        # Step 7: POST /spawn
        artifact_path = f"gen-{next_gen}"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=artifact_path,
            )
            log.info("spawned", component="prime", generation=next_gen,
                     container_id=spawn_result.get("container-id", ""))
        except Exception as e:
            log.error("spawn_failed", component="prime", error=str(e), generation=next_gen)
            consecutive_failures += 1
            total_generations += 1
            prime_module._status = "idle"
            continue

        # Step 8: Poll for outcome
        log.info("polling_for_outcome", component="prime", generation=next_gen)
        record: dict[str, Any] | None = None
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                versions = await supervisor.get_versions()
            except Exception as e:
                log.warning("poll_error", component="prime", error=str(e))
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

        # Step 9: Decide
        total_generations += 1
        viability = record.get("viability", {}) if record else {}
        status = viability.get("status", "non-viable")

        if status == "viable":
            log.info("generation_viable", component="prime", generation=next_gen)
            try:
                await supervisor.promote(next_gen)
            except Exception as e:
                log.error("promote_failed", component="prime", error=str(e))
            consecutive_failures = 0
            prime_module._status = "idle"
            break  # Done
        else:
            log.warning("generation_non_viable", component="prime", generation=next_gen,
                        failure_stage=viability.get("failure_stage", "unknown"))
            try:
                await supervisor.rollback(next_gen)
            except Exception as e:
                log.error("rollback_failed", component="prime", error=str(e))
            consecutive_failures += 1
            prime_module._status = "idle"

    log.info("generation_loop_done", component="prime", total_generations=total_generations,
             consecutive_failures=consecutive_failures)
