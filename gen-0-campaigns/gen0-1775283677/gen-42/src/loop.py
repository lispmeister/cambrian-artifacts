"""Generation loop — orchestrates the code generation lifecycle."""
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
    from src.manifest import build_manifest, write_manifest, extract_contracts_from_spec

    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    spec_path = Path(os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"))
    workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")

    supervisor = SupervisorClient(supervisor_url)

    gen_count = 0
    retry_count = 0
    failed_artifact_path: Path | None = None
    failed_diagnostics: dict[str, Any] | None = None

    while gen_count < max_gens and retry_count <= max_retries:
        import src.prime as prime_module
        prime_module._prime_status = "generating"

        try:
            # Step 1: Determine generation number
            versions = await supervisor.get_versions()
            if versions:
                next_gen = max(v.get("generation", 0) for v in versions) + 1
            else:
                next_gen = 1
            parent_gen = next_gen - 1

            log.info("generation_starting", component="prime",
                     generation=next_gen, retry_count=retry_count)

            # Step 2: Read spec
            spec_content = spec_path.read_text(encoding="utf-8")
            spec_hash = _compute_spec_hash(spec_path)

            # Step 3: Build prompt
            history_json = json.dumps(versions, indent=2)
            model = get_model(retry_count)

            if retry_count == 0 or failed_artifact_path is None:
                system_msg, user_msg = build_fresh_prompt(
                    spec_content, history_json, next_gen, parent_gen
                )
            else:
                failed_files = _read_failed_files(failed_artifact_path)
                system_msg, user_msg = build_retry_prompt(
                    spec_content, history_json, next_gen, parent_gen,
                    failed_diagnostics or {}, failed_files
                )

            # Step 4: Call LLM (with parse repair loop)
            parse_repair_count = 0
            raw_response: str | None = None
            files: dict[str, str] | None = None

            while True:
                if raw_response is None:
                    raw_response, token_usage = await call_llm(model, system_msg, user_msg)
                else:
                    # Parse repair
                    repair_system, repair_user = build_parse_repair_prompt(
                        str(last_parse_error), raw_response
                    )
                    raw_response, token_usage = await call_llm(model, repair_system, repair_user)

                try:
                    files = parse_files(raw_response)
                    break
                except ParseError as e:
                    last_parse_error = e
                    parse_repair_count += 1
                    log.warning("parse_error", component="prime",
                                generation=next_gen, attempt=parse_repair_count, error=str(e))
                    if parse_repair_count > max_parse_retries:
                        raise

            assert files is not None

            # Step 5 & 6: Write files to workspace
            artifact_dir = workspace / f"gen-{next_gen}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            for rel_path, content in files.items():
                file_path = artifact_dir / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

            # Copy spec file
            spec_dest_dir = artifact_dir / "spec"
            spec_dest_dir.mkdir(parents=True, exist_ok=True)
            spec_dest = spec_dest_dir / spec_path.name
            spec_dest.write_bytes(spec_path.read_bytes())

            # Extract contracts from spec
            contracts = extract_contracts_from_spec(spec_content)

            # Build file list
            file_list = _collect_file_list(artifact_dir)

            # Build and write manifest
            manifest = build_manifest(
                generation=next_gen,
                parent_generation=parent_gen,
                spec_hash=spec_hash,
                artifact_root=artifact_dir,
                files=file_list,
                model=model,
                token_usage=token_usage,
                contracts=contracts,
            )
            write_manifest(artifact_dir, manifest)

            # Step 7: Request verification
            prime_module._prime_status = "verifying"
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            log.info("spawn_requested", component="prime",
                     generation=next_gen, result=spawn_result)

            # Step 8: Poll for result
            record = await _poll_for_outcome(supervisor, next_gen)

            # Step 9: Decide
            viability = record.get("viability", {})
            status = viability.get("status", "non-viable")

            if status == "viable":
                await supervisor.promote(next_gen)
                log.info("generation_promoted", component="prime", generation=next_gen)
                prime_module._prime_status = "idle"
                return  # Done
            else:
                await supervisor.rollback(next_gen)
                log.warning("generation_rolled_back", component="prime", generation=next_gen)
                failed_artifact_path = artifact_dir
                failed_diagnostics = viability.get("diagnostics", {})
                retry_count += 1
                gen_count += 1
                prime_module._prime_status = "idle"

        except ParseError as e:
            log.error("parse_error_exhausted", component="prime", error=str(e))
            retry_count += 1
            gen_count += 1
            prime_module._prime_status = "idle"
        except Exception as exc:
            log.error("generation_error", component="prime", error=str(exc))
            await asyncio.sleep(5)
            gen_count += 1
            prime_module._prime_status = "idle"

    log.info("generation_loop_complete", component="prime",
             gen_count=gen_count, retry_count=retry_count)


async def _poll_for_outcome(supervisor: Any, generation: int) -> dict[str, Any]:
    """Poll until generation outcome is no longer in_progress."""
    while True:
        versions = await supervisor.get_versions()
        for record in versions:
            if record.get("generation") == generation:
                outcome = record.get("outcome", "in_progress")
                if outcome != "in_progress":
                    return record
        await asyncio.sleep(2)


def _compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of spec file."""
    import hashlib
    hasher = hashlib.sha256()
    hasher.update(spec_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def _collect_file_list(artifact_dir: Path) -> list[str]:
    """Collect all files in artifact directory as relative paths."""
    files = []
    for file_path in artifact_dir.rglob("*"):
        if file_path.is_file():
            rel = str(file_path.relative_to(artifact_dir))
            files.append(rel)
    return sorted(files)


def _read_failed_files(artifact_path: Path) -> dict[str, str]:
    """Read source files from a failed artifact directory."""
    files: dict[str, str] = {}
    if not artifact_path.exists():
        return files
    for file_path in artifact_path.rglob("*"):
        if file_path.is_file() and file_path.suffix in (".py", ".txt", ".md", ".json", ".toml"):
            try:
                rel = str(file_path.relative_to(artifact_path))
                files[rel] = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
    return files
