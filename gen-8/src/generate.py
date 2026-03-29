#!/usr/bin/env python3
"""LLM integration: prompt building, API calls, response parsing."""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import structlog
import anthropic

logger = structlog.get_logger().bind(component="prime")

SYSTEM_PROMPT = """\
You are a code generator. You produce complete, working Python codebases from specifications.

Rules:
- Output ONLY <file path="...">content</file:end> blocks. One block per file.
- Every file needed to build, test, and run the project must be in a <file> block.
- Include a requirements.txt with all dependencies.
- Include a test suite that exercises all functionality.
- The code must work in Python 3.14 inside a Docker container with a venv at /venv.
- Do NOT include manifest.json — it is generated separately.
- Do NOT include the spec file — it is copied separately.
- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use
  triple-quoted strings for multi-line content. Use \\n for embedded newlines in
  single-line strings. A bare newline inside "..." or '...' is a SyntaxError.
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings
  (r"...") or triple-quoted strings to avoid escaping issues.
- LLM API: MUST use `async with client.messages.stream(...) as stream:` with
  `await stream.get_final_message()`. Do NOT use `client.messages.create()` — the SDK
  raises an error for large max_tokens with non-streaming calls.
- aiohttp tests: use `aiohttp_client` pytest fixture. Do NOT use `AioHTTPTestCase` or
  `@unittest_run_loop` — both are deprecated and break in aiohttp 3.8+.
  Use `aiohttp_server` for mock servers. Set asyncio_mode = "auto" in pytest.ini.
"""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.
    Stops at the FIRST occurrence of </file:end> — whether on its own line
    or embedded within a line's content.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    END_TAG = "</file:end>"

    for line in response.splitlines(keepends=True):
        if current_path is None:
            m = re.match(r'<file path="([^"]+)">', line)
            if m:
                current_path = m.group(1)
                current_lines = []
        else:
            # Check if this line contains </file:end> anywhere
            stripped = line.rstrip("\n\r")
            if END_TAG in stripped:
                # Stop at the first occurrence of </file:end>
                # Only include content before the tag
                idx = stripped.index(END_TAG)
                before = stripped[:idx]
                if before:
                    current_lines.append(before + "\n")
                files[current_path] = "".join(current_lines)
                current_path = None
            else:
                current_lines.append(line)

    if current_path is not None:
        raise ParseError(f"Unclosed <file path={current_path!r}> block")

    return files


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation_number: int,
    parent_generation: int,
) -> str:
    history_json = json.dumps(generation_records, indent=2)
    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{history_json}\n\n"
        f"# Task\n\n"
        f"Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation_number}\n"
        f"Parent generation: {parent_generation}\n"
    )


def build_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation_number: int,
    parent_generation: int,
    failed_gen_number: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    history_json = json.dumps(generation_records, indent=2)
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "")
    diagnostics_json = json.dumps(diagnostics, indent=2)

    failed_code_parts: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = Path(file_path).suffix.lstrip(".")
        lang = ext if ext else "text"
        failed_code_parts.append(f"### {file_path}\n```{lang}\n{content}\n```")
    failed_code_str = "\n\n".join(failed_code_parts)

    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{history_json}\n\n"
        f"# Previous Attempt Failed\n\n"
        f"Generation {failed_gen_number} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        f"## Failed Source Code\n\n{failed_code_str}\n\n"
        f"## Diagnostics\n\n{diagnostics_json}\n\n"
        f"# Task\n\n"
        f"The previous attempt failed. Study the failed code and diagnostics above.\n"
        f"Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation_number}\n"
        f"Parent generation: {parent_generation}\n"
    )


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_llm_response: str,
) -> str:
    return (
        f"# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        f"# Malformed Response\n\n{raw_llm_response}\n\n"
        f"# Task\n\n"
        f"Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        f"matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )


def select_model(retry_count: int) -> str:
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    else:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


async def call_llm(
    user_message: str,
    model: str,
    api_key: str,
) -> tuple[str, int, int]:
    """
    Call LLM API with streaming. Returns (response_text, input_tokens, output_tokens).
    MUST use streaming — do NOT use client.messages.create() for large max_tokens.
    """
    client = anthropic.AsyncAnthropic(api_key=api_key)
    async with client.messages.stream(
        model=model,
        max_tokens=32768,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    response_text = "".join(
        block.text for block in message.content if hasattr(block, "text")
    )
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    return response_text, input_tokens, output_tokens


async def run_loop(set_status: Callable[[str], None]) -> None:
    """Main generation loop."""
    from src.supervisor import SupervisorClient
    from src.manifest import build_manifest, write_manifest, compute_spec_hash, extract_contracts_from_spec

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    supervisor_url = os.environ.get("CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400")
    spec_path_str = os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
    workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
    artifacts_root = Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", str(workspace)))
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))

    log = logger.bind()

    spec_path = Path(spec_path_str)
    if not spec_path.exists():
        log.error("Spec file not found", path=str(spec_path))
        return

    spec_content = spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(spec_path)
    log.info("Spec loaded", spec_hash=spec_hash)

    supervisor = SupervisorClient(supervisor_url)

    retry_count = 0
    gens_attempted = 0
    previous_failed_artifact_dir: Path | None = None
    previous_diagnostics: dict[str, Any] | None = None
    previous_gen_number: int | None = None

    while gens_attempted < max_gens and retry_count <= max_retries:
        set_status("generating")

        # Step 1: determine generation number
        try:
            records = await supervisor.get_versions()
        except Exception as e:
            log.warning("Failed to get versions from supervisor, assuming gen 1", error=str(e))
            records = []

        if records:
            generation_number = max(r.get("generation", 0) for r in records) + 1
        else:
            generation_number = 1

        parent_generation = generation_number - 1

        log.info("Starting generation", generation=generation_number, retry_count=retry_count)

        # Step 3: build prompt
        if retry_count > 0 and previous_diagnostics and previous_failed_artifact_dir and previous_gen_number:
            # Read failed files
            failed_files: dict[str, str] = {}
            if previous_failed_artifact_dir.exists():
                for f in previous_failed_artifact_dir.rglob("*"):
                    if f.is_file() and f.name != "manifest.json":
                        rel = str(f.relative_to(previous_failed_artifact_dir))
                        try:
                            failed_files[rel] = f.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            pass
            user_message = build_retry_prompt(
                spec_content=spec_content,
                generation_records=records,
                generation_number=generation_number,
                parent_generation=parent_generation,
                failed_gen_number=previous_gen_number,
                diagnostics=previous_diagnostics,
                failed_files=failed_files,
            )
        else:
            user_message = build_fresh_prompt(
                spec_content=spec_content,
                generation_records=records,
                generation_number=generation_number,
                parent_generation=parent_generation,
            )

        model = select_model(retry_count)
        log.info("Calling LLM", model=model, generation=generation_number)

        # Step 4: call LLM with parse repair loop
        raw_response: str = ""
        input_tokens = 0
        output_tokens = 0
        parsed_files: dict[str, str] | None = None
        parse_repair_count = 0

        try:
            raw_response, input_tokens, output_tokens = await call_llm(user_message, model, api_key)
        except Exception as e:
            log.error("LLM call failed", error=str(e), generation=generation_number)
            retry_count += 1
            gens_attempted += 1
            continue

        # Step 5: parse response
        while True:
            try:
                parsed_files = parse_files(raw_response)
                break
            except ParseError as pe:
                log.warning("ParseError", error=str(pe), repair_attempt=parse_repair_count)
                if parse_repair_count >= max_parse_retries:
                    log.error("Max parse retries exhausted", generation=generation_number)
                    parsed_files = None
                    break
                # Attempt parse repair
                repair_prompt = build_parse_repair_prompt(str(pe), raw_response)
                try:
                    raw_response, it, ot = await call_llm(repair_prompt, model, api_key)
                    input_tokens += it
                    output_tokens += ot
                except Exception as e2:
                    log.error("LLM repair call failed", error=str(e2))
                    parsed_files = None
                    break
                parse_repair_count += 1

        if parsed_files is None:
            retry_count += 1
            gens_attempted += 1
            continue

        # Step 6: write files to workspace
        artifact_dir = artifacts_root / f"gen-{generation_number}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in parsed_files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_dest = spec_dest_dir / spec_path.name
        spec_dest.write_bytes(spec_path.read_bytes())

        # Gather file list
        all_files: list[str] = []
        for f in artifact_dir.rglob("*"):
            if f.is_file() and f.name != "manifest.json":
                all_files.append(str(f.relative_to(artifact_dir)))
        all_files.append("manifest.json")

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Build manifest
        manifest_data = build_manifest(
            artifact_root=artifact_dir,
            files=all_files,
            generation=generation_number,
            parent_generation=parent_generation,
            spec_hash=spec_hash,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest_data)

        log.info("Artifact written", generation=generation_number, files=len(all_files))

        # Step 7: POST /spawn
        set_status("verifying")
        artifact_path = f"gen-{generation_number}"
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=generation_number,
                artifact_path=artifact_path,
            )
            log.info("Spawn requested", result=spawn_result, generation=generation_number)
        except Exception as e:
            log.error("Spawn failed", error=str(e), generation=generation_number)
            retry_count += 1
            gens_attempted += 1
            continue

        # Step 8: poll until outcome != in_progress
        log.info("Polling for outcome", generation=generation_number)
        max_poll = 600
        poll_count = 0
        outcome_record: dict[str, Any] | None = None
        while poll_count < max_poll:
            await asyncio.sleep(2)
            poll_count += 1
            try:
                versions = await supervisor.get_versions()
            except Exception as e:
                log.warning("Poll failed", error=str(e))
                continue
            for rec in versions:
                if rec.get("generation") == generation_number:
                    if rec.get("outcome") != "in_progress":
                        outcome_record = rec
                    break
            if outcome_record is not None:
                break

        if outcome_record is None:
            log.error("Timed out waiting for outcome", generation=generation_number)
            retry_count += 1
            gens_attempted += 1
            continue

        # Step 9: decide
        viability = outcome_record.get("viability", {})
        status = viability.get("status", "non-viable") if viability else "non-viable"

        if status == "viable":
            log.info("Generation viable — promoting", generation=generation_number)
            try:
                await supervisor.promote(generation_number)
            except Exception as e:
                log.error("Promote failed", error=str(e))
            set_status("idle")
            return
        else:
            log.info("Generation non-viable — rolling back", generation=generation_number)
            previous_diagnostics = viability.get("diagnostics") if viability else None
            previous_failed_artifact_dir = artifact_dir
            previous_gen_number = generation_number
            try:
                await supervisor.rollback(generation_number)
            except Exception as e:
                log.error("Rollback failed", error=str(e))
            retry_count += 1
            gens_attempted += 1

    log.error("Generation loop exhausted", gens_attempted=gens_attempted, retry_count=retry_count)
    set_status("idle")