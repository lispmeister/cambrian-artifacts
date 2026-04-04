"""Generation loop — the core Prime lifecycle."""

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
        "max_retries": int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3")),
        "max_gens": int(os.environ.get("CAMBRIAN_MAX_GENS", "5")),
        "max_parse_retries": int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2")),
        "supervisor_url": os.environ.get(
            "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
        ),
        "spec_path": os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"),
        "workspace": os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"),
        "artifacts_root": os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace"),
        "token_budget": int(os.environ.get("CAMBRIAN_TOKEN_BUDGET", "0")),
    }


async def generation_loop() -> None:
    """Main generation loop."""
    import prime as prime_module  # noqa: F401 — update status

    from src.supervisor import SupervisorClient
    from src.generate import (
        ParseError,
        build_fresh_prompt,
        build_retry_prompt,
        build_repair_prompt,
        call_llm,
        parse_files,
        get_model,
    )
    from src.manifest import (
        build_manifest,
        write_manifest,
        compute_spec_hash,
        extract_contracts_from_spec,
    )

    config = get_config()
    spec_path = Path(config["spec_path"])
    workspace = Path(config["workspace"])
    supervisor_url = config["supervisor_url"]
    max_retries = config["max_retries"]
    max_gens = config["max_gens"]
    max_parse_retries = config["max_parse_retries"]

    supervisor = SupervisorClient(supervisor_url)

    # Read spec
    if not spec_path.exists():
        log.error("spec_not_found", component="prime", path=str(spec_path))
        return

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)

    log.info(
        "spec_loaded",
        component="prime",
        spec_hash=spec_hash,
        spec_path=str(spec_path),
    )

    # Get generation history
    history = await supervisor.get_versions()

    # Determine next generation number
    if history:
        next_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        next_gen = 1

    own_generation = int(os.environ.get("CAMBRIAN_GENERATION", "0"))
    parent_gen = next_gen - 1

    retry_count = 0
    gen_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        gen_num = next_gen + gen_count
        artifact_dir = workspace / f"gen-{gen_num}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "generation_starting",
            component="prime",
            generation=gen_num,
            retry_count=retry_count,
        )

        # Build prompt
        if retry_count == 0 or failed_diagnostics is None:
            messages = build_fresh_prompt(spec_content, history, gen_num, parent_gen)
        else:
            # Read failed source files
            failed_files: dict[str, str] = {}
            if failed_artifact_path and failed_artifact_path.exists():
                for fpath in failed_artifact_path.rglob("*"):
                    if fpath.is_file() and fpath.name != "manifest.json":
                        rel = fpath.relative_to(failed_artifact_path)
                        try:
                            failed_files[str(rel)] = fpath.read_text(encoding="utf-8")
                        except Exception:
                            pass
            messages = build_retry_prompt(
                spec_content,
                history,
                gen_num,
                parent_gen,
                failed_diagnostics,
                failed_files,
            )

        # Get model
        model = get_model(retry_count)

        # Call LLM with parse repair loop
        parse_retry = 0
        parsed_files: dict[str, str] | None = None
        raw_response = ""

        while parse_retry <= max_parse_retries:
            try:
                raw_response = await call_llm(model, messages)
                parsed_files = parse_files(raw_response)
                break
            except ParseError as exc:
                parse_retry += 1
                log.warning(
                    "parse_error",
                    component="prime",
                    generation=gen_num,
                    parse_retry=parse_retry,
                    error=str(exc),
                )
                if parse_retry > max_parse_retries:
                    break
                # Build repair prompt
                messages = build_repair_prompt(str(exc), raw_response)

        if parsed_files is None:
            log.error(
                "parse_failed_all_retries",
                component="prime",
                generation=gen_num,
            )
            retry_count += 1
            gen_count += 1
            continue

        # Write files to artifact directory
        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Compute file list
        all_files: list[str] = []
        for fpath in sorted(artifact_dir.rglob("*")):
            if fpath.is_file():
                rel = str(fpath.relative_to(artifact_dir))
                all_files.append(rel)

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Token usage (will be populated from LLM response)
        token_usage = {"input": 0, "output": 0}

        # Build and write manifest
        manifest_data = build_manifest(
            generation=gen_num,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_root=artifact_dir,
            files=all_files,
            producer_model=model,
            token_usage=token_usage,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest_data)

        # Request verification
        artifact_rel_path = f"gen-{gen_num}"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=gen_num,
                artifact_path=artifact_rel_path,
            )
            log.info(
                "spawn_requested",
                component="prime",
                generation=gen_num,
                result=spawn_result,
            )
        except Exception as exc:
            log.error(
                "spawn_failed",
                component="prime",
                generation=gen_num,
                error=str(exc),
            )
            retry_count += 1
            gen_count += 1
            continue

        # Poll for result
        outcome = "in_progress"
        viability: dict[str, Any] | None = None
        poll_count = 0

        while outcome == "in_progress":
            await asyncio.sleep(2)
            poll_count += 1
            try:
                versions = await supervisor.get_versions()
                for record in versions:
                    if record.get("generation") == gen_num:
                        outcome = record.get("outcome", "in_progress")
                        viability = record.get("viability")
                        break
            except Exception as exc:
                log.warning(
                    "poll_error",
                    component="prime",
                    generation=gen_num,
                    error=str(exc),
                )
                await asyncio.sleep(2)

            if poll_count > 300:  # 10 minutes timeout
                log.error("poll_timeout", component="prime", generation=gen_num)
                break

        # Decide: promote or rollback
        if viability and viability.get("status") == "viable":
            try:
                await supervisor.promote(gen_num)
                log.info(
                    "generation_promoted",
                    component="prime",
                    generation=gen_num,
                )
            except Exception as exc:
                log.error(
                    "promote_failed",
                    component="prime",
                    generation=gen_num,
                    error=str(exc),
                )
            return  # Done — one successful promotion per loop
        else:
            try:
                await supervisor.rollback(gen_num)
                log.info(
                    "generation_rolled_back",
                    component="prime",
                    generation=gen_num,
                )
            except Exception as exc:
                log.error(
                    "rollback_failed",
                    component="prime",
                    generation=gen_num,
                    error=str(exc),
                )

            # Store failure context for retry
            failed_artifact_path = artifact_dir
            failed_diagnostics = viability.get("diagnostics") if viability else None
            retry_count += 1
            gen_count += 1

            if retry_count > max_retries:
                log.error(
                    "max_retries_reached",
                    component="prime",
                    generation=gen_num,
                    max_retries=max_retries,
                )
                return

    log.info("generation_loop_complete", component="prime")
