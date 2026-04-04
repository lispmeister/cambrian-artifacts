"""Generation loop — orchestrates LLM calls, file writing, and Supervisor interaction."""
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
    from src.manifest import build_manifest, write_manifest, compute_spec_hash

    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    spec_path = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    supervisor_url = os.environ.get(
        "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
    )

    spec_file = Path(spec_path)
    if not spec_file.exists():
        log.warning(
            "spec_not_found",
            component="prime",
            path=str(spec_file),
        )
        return

    spec_content = spec_file.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_file)

    supervisor = SupervisorClient(supervisor_url)

    generation_count = 0
    retry_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while generation_count < max_gens and retry_count <= max_retries:
        # Step 1: determine generation number
        try:
            versions = await supervisor.get_versions()
        except Exception as exc:
            log.error(
                "supervisor_unreachable",
                component="prime",
                error=str(exc),
            )
            await asyncio.sleep(5)
            continue

        if versions:
            next_gen = max(r.get("generation", 0) for r in versions) + 1
        else:
            next_gen = 1

        parent_gen = next_gen - 1

        log.info(
            "generation_starting",
            component="prime",
            generation=next_gen,
            retry_count=retry_count,
        )

        # Step 3: build prompt
        history_json = json.dumps(versions, indent=2)
        if failed_artifact_path is not None and failed_diagnostics is not None:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path.exists():
                for fpath in sorted(failed_artifact_path.rglob("*")):
                    if fpath.is_file() and fpath.name != "manifest.json":
                        rel = fpath.relative_to(failed_artifact_path)
                        try:
                            failed_files[str(rel)] = fpath.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
            prompt = build_retry_prompt(
                spec_content=spec_content,
                history_json=history_json,
                generation=next_gen,
                parent=parent_gen,
                failed_generation=next_gen - 1,
                diagnostics=failed_diagnostics,
                failed_files=failed_files,
            )
        else:
            prompt = build_fresh_prompt(
                spec_content=spec_content,
                history_json=history_json,
                generation=next_gen,
                parent=parent_gen,
            )

        # Step 4: call LLM with parse repair loop
        model = get_model(retry_count)
        raw_response: str | None = None
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        while parsed_files is None:
            try:
                raw_response = await call_llm(
                    system_prompt=prompt["system"],
                    user_message=prompt["user"],
                    model=model,
                )
                parsed_files = parse_files(raw_response)
            except ParseError as exc:
                parse_repair_count += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    generation=next_gen,
                    attempt=parse_repair_count,
                    error=str(exc),
                )
                if parse_repair_count > max_parse_retries:
                    break
                # Try repair
                repair_prompt = build_parse_repair_prompt(
                    parse_error_message=str(exc),
                    raw_response=raw_response or "",
                )
                try:
                    raw_response = await call_llm(
                        system_prompt=repair_prompt["system"],
                        user_message=repair_prompt["user"],
                        model=model,
                    )
                    parsed_files = parse_files(raw_response)
                except ParseError:
                    parsed_files = None

        if parsed_files is None:
            log.error(
                "parse_failed_all_retries",
                component="prime",
                generation=next_gen,
            )
            generation_count += 1
            retry_count += 1
            failed_artifact_path = None
            failed_diagnostics = None
            continue

        # Step 6: write files to workspace
        artifact_dir = workspace / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_file.name
        spec_dest.write_bytes(spec_file.read_bytes())

        # Get token usage from LLM (stored via call_llm)
        from src.generate import _last_token_usage
        token_usage = _last_token_usage.copy()

        # Build file list
        all_files = []
        for fpath in sorted(artifact_dir.rglob("*")):
            if fpath.is_file():
                rel = str(fpath.relative_to(artifact_dir))
                all_files.append(rel)

        # Build manifest
        from src.generate import get_producer_model
        manifest = build_manifest(
            artifact_root=artifact_dir,
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            files=all_files,
            producer_model=get_producer_model(retry_count),
            token_usage=token_usage,
            spec_content=spec_content,
        )

        write_manifest(artifact_dir, manifest)

        log.info(
            "artifact_written",
            component="prime",
            generation=next_gen,
            path=str(artifact_dir),
        )

        # Step 7: POST /spawn
        artifact_path = f"gen-{next_gen}"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=artifact_path,
            )
            log.info(
                "spawn_requested",
                component="prime",
                generation=next_gen,
                result=spawn_result,
            )
        except Exception as exc:
            log.error(
                "spawn_failed",
                component="prime",
                generation=next_gen,
                error=str(exc),
            )
            generation_count += 1
            retry_count += 1
            continue

        # Step 8: poll until outcome != in_progress
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
            except Exception as exc:
                log.warning(
                    "poll_error",
                    component="prime",
                    generation=next_gen,
                    error=str(exc),
                )
                continue

            record = next(
                (r for r in versions if r.get("generation") == next_gen), None
            )
            if record is None:
                continue

            outcome = record.get("outcome", "in_progress")
            if outcome != "in_progress":
                log.info(
                    "generation_outcome",
                    component="prime",
                    generation=next_gen,
                    outcome=outcome,
                )
                break

        # Step 9: decide
        viability = record.get("viability", {})
        status = viability.get("status", "non-viable")

        if status == "viable":
            try:
                await supervisor.promote(next_gen)
                log.info(
                    "generation_promoted",
                    component="prime",
                    generation=next_gen,
                )
            except Exception as exc:
                log.error(
                    "promote_failed",
                    component="prime",
                    generation=next_gen,
                    error=str(exc),
                )
            # Done — one successful promotion
            return
        else:
            # Rollback
            try:
                await supervisor.rollback(next_gen)
                log.info(
                    "generation_rolled_back",
                    component="prime",
                    generation=next_gen,
                )
            except Exception as exc:
                log.error(
                    "rollback_failed",
                    component="prime",
                    generation=next_gen,
                    error=str(exc),
                )

            generation_count += 1
            retry_count += 1

            diagnostics = viability.get("diagnostics")
            if diagnostics:
                failed_artifact_path = artifact_dir
                failed_diagnostics = diagnostics
            else:
                failed_artifact_path = None
                failed_diagnostics = None

            if retry_count > max_retries:
                log.error(
                    "max_retries_exhausted",
                    component="prime",
                    generation=next_gen,
                    max_retries=max_retries,
                )
                return

    log.info(
        "generation_loop_complete",
        component="prime",
        generation_count=generation_count,
    )
