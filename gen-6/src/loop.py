#!/usr/bin/env python3
"""Generation loop logic."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import structlog

logger = structlog.get_logger().bind(component="prime")

WORKSPACE = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
ARTIFACTS_ROOT = Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", str(WORKSPACE)))
SPEC_PATH = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
POLL_INTERVAL = float(os.environ.get("CAMBRIAN_POLL_INTERVAL", "2"))


async def run_generation_loop(set_status: Callable[[str], None]) -> None:
    """Main generation loop."""
    from src.supervisor import SupervisorClient
    from src.generate import (
        build_fresh_prompt,
        build_retry_prompt,
        build_parse_repair_prompt,
        call_llm,
        parse_files,
        ParseError,
        SYSTEM_PROMPT,
    )
    from src.manifest import build_manifest, write_manifest, extract_contracts_from_spec

    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )
    supervisor = SupervisorClient(supervisor_url)

    retry_count = 0
    gens_produced = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gens_produced < MAX_GENS and retry_count <= MAX_RETRIES:
        set_status("generating")

        # Step 1: Determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as e:
            logger.error("failed to get versions", error=str(e))
            await asyncio.sleep(5)
            continue

        if versions:
            next_gen = max(v.get("generation", 0) for v in versions) + 1
            parent_gen = max(v.get("generation", 0) for v in versions)
        else:
            next_gen = 1
            parent_gen = 0

        log = logger.bind(generation=next_gen)
        log.info("starting generation", retry_count=retry_count)

        # Step 2: Read spec
        try:
            spec_content = SPEC_PATH.read_text(encoding="utf-8")
        except Exception as e:
            log.error("failed to read spec", error=str(e))
            await asyncio.sleep(5)
            continue

        # Step 3+4: Build prompt and call LLM (with parse repair loop)
        history_json = json.dumps(versions, indent=2)

        if retry_count == 0 or failed_diagnostics is None:
            system_msg, user_msg = build_fresh_prompt(
                spec_content=spec_content,
                generation_records_json=history_json,
                generation=next_gen,
                parent=parent_gen,
            )
        else:
            # Informed retry
            failed_files: dict[str, str] = {}
            if failed_artifact_path is not None and failed_artifact_path.exists():
                for fp in failed_artifact_path.rglob("*"):
                    if fp.is_file() and fp.name != "manifest.json":
                        rel = str(fp.relative_to(failed_artifact_path))
                        try:
                            failed_files[rel] = fp.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
            system_msg, user_msg = build_retry_prompt(
                spec_content=spec_content,
                generation_records_json=history_json,
                generation=next_gen,
                parent=parent_gen,
                failed_generation=next_gen - 1,
                diagnostics=failed_diagnostics,
                failed_files=failed_files,
            )

        # LLM call with parse repair
        raw_response: str = ""
        parsed_files: dict[str, str] = {}
        parse_repair_count = 0
        llm_input_tokens = 0
        llm_output_tokens = 0
        model_used = ""
        llm_success = False

        while True:
            try:
                log.info("calling LLM", parse_repair_count=parse_repair_count)
                raw_response, usage, model_used = await call_llm(
                    system_msg=system_msg,
                    user_msg=user_msg,
                    retry_count=retry_count,
                )
                llm_input_tokens = usage.get("input", 0)
                llm_output_tokens = usage.get("output", 0)

                try:
                    parsed_files = parse_files(raw_response)
                    llm_success = True
                    break
                except ParseError as pe:
                    log.warning("parse error", error=str(pe), repair_attempt=parse_repair_count)
                    if parse_repair_count >= MAX_PARSE_RETRIES:
                        log.error("max parse retries reached")
                        break
                    # Attempt repair
                    _, repair_user_msg = build_parse_repair_prompt(
                        parse_error_message=str(pe),
                        raw_response=raw_response,
                    )
                    system_msg = SYSTEM_PROMPT
                    user_msg = repair_user_msg
                    parse_repair_count += 1

            except Exception as e:
                log.error("LLM call failed", error=str(e))
                await asyncio.sleep(5)
                break

        if not llm_success:
            log.warning("generation failed (parse/LLM error)", retry_count=retry_count)
            retry_count += 1
            gens_produced += 1
            continue

        # Step 6: Write files to workspace
        artifact_dir = ARTIFACTS_ROOT / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = SPEC_PATH.name
        shutil.copy2(SPEC_PATH, spec_dest_dir / spec_filename)

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Build file list
        all_files: list[str] = []
        spec_rel = f"spec/{spec_filename}"
        for fp in artifact_dir.rglob("*"):
            if fp.is_file():
                rel = str(fp.relative_to(artifact_dir))
                all_files.append(rel)

        if spec_rel not in all_files:
            all_files.append(spec_rel)

        # Write manifest
        manifest_data = build_manifest(
            artifact_root=artifact_dir,
            generation=next_gen,
            parent_generation=parent_gen,
            spec_path=artifact_dir / "spec" / spec_filename,
            files=[f for f in all_files if f != "manifest.json"],
            model=model_used,
            token_input=llm_input_tokens,
            token_output=llm_output_tokens,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest_data)
        all_files_with_manifest = [f for f in all_files if f != "manifest.json"]
        all_files_with_manifest.append("manifest.json")

        log.info("artifact written", path=str(artifact_dir), files=len(all_files_with_manifest))

        # Step 7: POST /spawn
        set_status("verifying")
        try:
            spawn_resp = await supervisor.spawn(
                spec_hash=manifest_data["spec-hash"],
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn response", response=spawn_resp)
        except Exception as e:
            log.error("spawn failed", error=str(e))
            retry_count += 1
            gens_produced += 1
            continue

        # Step 8: Poll until outcome != in_progress
        log.info("polling for viability result")
        viability: dict[str, Any] | None = None
        while True:
            await asyncio.sleep(POLL_INTERVAL)
            try:
                versions_now = await supervisor.get_versions()
            except Exception as e:
                log.warning("poll get_versions failed", error=str(e))
                continue
            record = next(
                (v for v in versions_now if v.get("generation") == next_gen), None
            )
            if record is None:
                continue
            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                viability = record.get("viability")
                log.info("got outcome", outcome=outcome)
                break

        # Step 9: Decide
        if viability and viability.get("status") == "viable":
            log.info("generation viable — promoting")
            try:
                await supervisor.promote(next_gen)
            except Exception as e:
                log.error("promote failed", error=str(e))
            log.info("promoted", generation=next_gen)
            retry_count = 0
            failed_artifact_path = None
            failed_diagnostics = None
        else:
            log.warning("generation non-viable — rolling back")
            try:
                await supervisor.rollback(next_gen)
            except Exception as e:
                log.error("rollback failed", error=str(e))
            failed_artifact_path = artifact_dir
            failed_diagnostics = (
                viability.get("diagnostics") if viability else None
            )
            retry_count += 1

        gens_produced += 1
        set_status("idle")

        if retry_count > MAX_RETRIES:
            log.error("max retries exhausted — stopping")
            break

    log.info("generation loop finished", gens_produced=gens_produced, retry_count=retry_count)
    set_status("idle")