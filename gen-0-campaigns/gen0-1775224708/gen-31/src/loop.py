"""Generation loop — orchestrates LLM calls, file writing, and Supervisor interaction."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()


def get_config() -> dict[str, Any]:
    """Read configuration from environment variables."""
    return {
        "model": os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6"),
        "escalation_model": os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6"),
        "max_gens": int(os.environ.get("CAMBRIAN_MAX_GENS", "5")),
        "max_retries": int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3")),
        "max_parse_retries": int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2")),
        "supervisor_url": os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"),
        "spec_path": os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"),
        "workspace": os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"),
        "artifacts_root": os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace"),
        "token_budget": int(os.environ.get("CAMBRIAN_TOKEN_BUDGET", "0")),
    }


async def generation_loop() -> None:
    """Main generation loop."""
    import src.prime as prime_module
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
    from src.manifest import (
        build_manifest,
        compute_artifact_hash,
        compute_spec_hash,
        write_manifest,
    )

    config = get_config()
    spec_path = Path(config["spec_path"])
    workspace = Path(config["workspace"])
    artifacts_root = Path(config["artifacts_root"])

    # Read spec
    if not spec_path.exists():
        log.error("spec_not_found", component="prime", path=str(spec_path))
        return

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)

    log.info("spec_loaded", component="prime", spec_path=str(spec_path), spec_hash=spec_hash)

    supervisor = SupervisorClient(config["supervisor_url"])

    retry_count = 0
    gen_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < config["max_gens"] and retry_count <= config["max_retries"]:
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

        log.info("starting_generation", component="prime", generation=next_gen, retry_count=retry_count)
        prime_module._status = "generating"

        # Step 3: Build prompt
        model = get_model(config["model"], config["escalation_model"], retry_count)

        if retry_count > 0 and failed_artifact_path is not None and failed_diagnostics is not None:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path.exists():
                for file_path in failed_artifact_path.rglob("*"):
                    if file_path.is_file() and file_path.name != "manifest.json":
                        rel = str(file_path.relative_to(failed_artifact_path))
                        try:
                            failed_files[rel] = file_path.read_text(encoding="utf-8")
                        except Exception:
                            pass

            system_msg, user_msg = build_informed_retry_prompt(
                spec_content=spec_content,
                generation_records=history,
                generation=next_gen,
                parent=parent_gen,
                failed_generation=next_gen - 1,
                diagnostics=failed_diagnostics,
                failed_files=failed_files,
            )
        else:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=history,
                generation=next_gen,
                parent=parent_gen,
            )

        # Step 4: Call LLM with parse repair loop
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0
        parse_error_msg: str = ""

        try:
            raw_response = await call_llm(model=model, system=system_msg, user=user_msg)
        except Exception as e:
            log.error("llm_call_failed", component="prime", error=str(e))
            retry_count += 1
            gen_count += 1
            await asyncio.sleep(2)
            continue

        # Step 5: Parse response with repair loop
        while True:
            try:
                parsed_files = parse_files(raw_response)
                break
            except ParseError as pe:
                parse_error_msg = str(pe)
                log.warning("parse_error", component="prime", error=parse_error_msg,
                            repair_attempt=parse_repair_count)

                if parse_repair_count >= config["max_parse_retries"]:
                    log.error("parse_repair_exhausted", component="prime",
                              max_parse_retries=config["max_parse_retries"])
                    parsed_files = None
                    break

                # Attempt parse repair
                _, repair_user = build_parse_repair_prompt(
                    parse_error_message=parse_error_msg,
                    raw_response=raw_response,
                )
                try:
                    raw_response = await call_llm(model=model, system=system_msg, user=repair_user)
                    parse_repair_count += 1
                except Exception as e:
                    log.error("repair_llm_failed", component="prime", error=str(e))
                    parsed_files = None
                    break

        if parsed_files is None:
            log.error("parse_failed_all_retries", component="prime", generation=next_gen)
            retry_count += 1
            gen_count += 1
            continue

        # Step 6: Write files to workspace
        artifact_dir = artifacts_root / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file to artifact
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Collect file list
        files: list[str] = []
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                files.append(str(fp.relative_to(artifact_dir)))
        files.sort()

        # Compute artifact hash (before writing manifest)
        artifact_hash = compute_artifact_hash(artifact_dir, files)

        # Get token usage from last LLM call (stored as module-level state)
        from src import generate as gen_module
        token_usage = gen_module._last_token_usage

        # Build and write manifest
        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_hash=artifact_hash,
            producer_model=model,
            token_usage=token_usage,
            files=files + ["manifest.json"],
            spec_content=spec_content,
        )

        write_manifest(artifact_dir, manifest)

        log.info("artifact_written", component="prime", generation=next_gen,
                 artifact_dir=str(artifact_dir), files=len(files))

        # Step 7: Request verification
        prime_module._status = "verifying"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn_requested", component="prime", generation=next_gen,
                     result=spawn_result)
        except Exception as e:
            log.error("spawn_failed", component="prime", error=str(e))
            retry_count += 1
            gen_count += 1
            await asyncio.sleep(5)
            continue

        # Step 8: Poll until outcome != in_progress
        viability: dict[str, Any] | None = None
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
                record = next(
                    (r for r in versions if r.get("generation") == next_gen), None
                )
                if record is None:
                    continue
                outcome = record.get("outcome", "in_progress")
                if outcome != "in_progress":
                    viability = record.get("viability")
                    break
            except Exception as e:
                log.warning("poll_error", component="prime", error=str(e))
                await asyncio.sleep(5)

        # Step 9: Decide
        if viability and viability.get("status") == "viable":
            log.info("generation_viable", component="prime", generation=next_gen)
            try:
                await supervisor.promote(generation=next_gen)
                log.info("promoted", component="prime", generation=next_gen)
            except Exception as e:
                log.error("promote_failed", component="prime", error=str(e))
            prime_module._status = "idle"
            return  # Done — one successful promotion per loop
        else:
            log.warning("generation_non_viable", component="prime", generation=next_gen)
            failed_artifact_path = artifact_dir
            failed_diagnostics = viability.get("diagnostics") if viability else None

            try:
                await supervisor.rollback(generation=next_gen)
                log.info("rolled_back", component="prime", generation=next_gen)
            except Exception as e:
                log.error("rollback_failed", component="prime", error=str(e))

            retry_count += 1
            gen_count += 1

            if retry_count > config["max_retries"]:
                log.error("max_retries_exhausted", component="prime",
                          max_retries=config["max_retries"])
                break

    prime_module._status = "idle"
    log.info("generation_loop_complete", component="prime",
             gen_count=gen_count, retry_count=retry_count)
