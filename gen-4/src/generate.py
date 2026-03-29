"""LLM integration: prompt building, API calls, response parsing."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(component="prime")

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """Raised when LLM response cannot be parsed."""
    pass


# ---------------------------------------------------------------------------
# Response parser — state machine (NOT dotall regex)
# ---------------------------------------------------------------------------


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    for line in response.splitlines(keepends=True):
        if current_path is None:
            m = re.match(r'<file path="([^"]+)">', line)
            if m:
                current_path = m.group(1)
                current_lines = []
        elif line.rstrip("\n\r") == "</file:end>":
            files[current_path] = "".join(current_lines)
            current_path = None
        else:
            current_lines.append(line)

    if current_path is not None:
        raise ParseError(f"Unclosed <file path={current_path!r}> block")

    return files


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

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
  Use `aiohttp_server` for mock servers. Set asyncio_mode = "auto" in pytest.ini.\
"""


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> str:
    records_json = json.dumps(generation_records, indent=2)
    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{records_json}\n\n"
        f"# Task\n\n"
        f"Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    records_json = json.dumps(generation_records, indent=2)
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "")
    diag_json = json.dumps(diagnostics, indent=2)

    # Build failed source code section
    file_sections = []
    for file_path, content in failed_files.items():
        ext = Path(file_path).suffix.lstrip(".")
        lang = ext if ext else "text"
        file_sections.append(f"### {file_path}\n```{lang}\n{content}\n```")
    failed_source = "\n\n".join(file_sections)

    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{records_json}\n\n"
        f"# Previous Attempt Failed\n\n"
        f"Generation {failed_generation} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        f"## Failed Source Code\n\n{failed_source}\n\n"
        f"## Diagnostics\n\n{diag_json}\n\n"
        f"# Task\n\n"
        f"The previous attempt failed. Study the failed code and diagnostics above.\n"
        f"Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
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


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------


async def call_llm(
    *,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 16000,
) -> tuple[str, int, int]:
    """
    Call the Anthropic LLM using streaming.
    Returns (response_text, input_tokens, output_tokens).
    """
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )

    log.info("Calling LLM", model=model, max_tokens=max_tokens)

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = await stream.get_final_message()

    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text += block.text

    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens

    log.info(
        "LLM response received",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        response_length=len(text),
    )

    return text, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------


class GenerationLoop:
    """Orchestrates the full generation loop."""

    def __init__(self) -> None:
        self.spec_path = Path(
            os.environ.get("CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md")
        )
        self.supervisor_url = os.environ.get(
            "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
        )
        self.model = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
        self.escalation_model = os.environ.get(
            "CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6"
        )
        self.max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        self.max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        self.max_parse_retries = int(
            os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2")
        )
        self.workspace = Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace"))
        self.artifacts_root = Path(
            os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", str(self.workspace))
        )

    async def run(self) -> None:
        """Run the generation loop."""
        from src.supervisor import SupervisorClient
        from src.manifest import (
            build_manifest,
            compute_spec_hash,
            write_manifest,
            extract_contracts_from_spec,
        )

        log.info("Generation loop starting")

        supervisor = SupervisorClient(self.supervisor_url)
        retry_count = 0
        gens_attempted = 0

        while gens_attempted < self.max_gens:
            # Step 1: determine generation number
            try:
                records = await supervisor.get_versions()
            except Exception as e:
                log.error("Failed to get versions from supervisor", error=str(e))
                await asyncio.sleep(5)
                continue

            max_gen = max((r.get("generation", 0) for r in records), default=0)
            gen_number = max_gen + 1
            parent_gen = max_gen

            # Step 2: read spec
            if not self.spec_path.exists():
                log.error("Spec file not found", path=str(self.spec_path))
                return

            spec_content = self.spec_path.read_text()
            spec_hash = compute_spec_hash(self.spec_path)

            # Choose model
            model = self.model if retry_count == 0 else self.escalation_model

            log.info(
                "Starting generation",
                generation=gen_number,
                parent=parent_gen,
                retry_count=retry_count,
                model=model,
            )

            # Step 3+4: build prompt and call LLM with parse retry
            user_prompt: str
            if retry_count == 0:
                user_prompt = build_fresh_prompt(
                    spec_content=spec_content,
                    generation_records=records,
                    generation=gen_number,
                    parent=parent_gen,
                )
            else:
                # Find last failed record for diagnostics
                failed_gen = gen_number - 1
                failed_record = next(
                    (r for r in reversed(records) if r.get("generation") == failed_gen),
                    None,
                )
                diagnostics: dict[str, Any] = {}
                failed_files: dict[str, str] = {}

                if failed_record and failed_record.get("viability"):
                    viability = failed_record["viability"]
                    if viability.get("diagnostics"):
                        diagnostics = viability["diagnostics"]

                # Read failed files from disk
                failed_artifact = self.artifacts_root / f"gen-{failed_gen}"
                if failed_artifact.exists():
                    for p in failed_artifact.rglob("*"):
                        if p.is_file() and p.name != "manifest.json":
                            rel = str(p.relative_to(failed_artifact))
                            try:
                                failed_files[rel] = p.read_text()
                            except Exception:
                                pass

                user_prompt = build_retry_prompt(
                    spec_content=spec_content,
                    generation_records=records,
                    generation=gen_number,
                    parent=parent_gen,
                    failed_generation=failed_gen,
                    diagnostics=diagnostics,
                    failed_files=failed_files,
                )

            # Call LLM with parse repair loop
            parsed_files: dict[str, str] | None = None
            token_input = 0
            token_output = 0

            raw_response = ""
            parse_retries = 0

            while True:
                try:
                    raw_response, ti, to = await call_llm(
                        system=SYSTEM_PROMPT,
                        user=user_prompt,
                        model=model,
                    )
                    token_input += ti
                    token_output += to
                except Exception as e:
                    log.error("LLM call failed", error=str(e))
                    await asyncio.sleep(5)
                    break

                try:
                    parsed_files = parse_files(raw_response)
                    break  # success
                except ParseError as pe:
                    if parse_retries < self.max_parse_retries:
                        parse_retries += 1
                        log.warning(
                            "Parse error, attempting repair",
                            error=str(pe),
                            attempt=parse_retries,
                        )
                        user_prompt = build_parse_repair_prompt(
                            parse_error_message=str(pe),
                            raw_llm_response=raw_response,
                        )
                    else:
                        log.error("All parse repairs failed", error=str(pe))
                        parsed_files = None
                        break

            if parsed_files is None:
                log.error("Failed to get parseable LLM output")
                retry_count += 1
                gens_attempted += 1
                if retry_count > self.max_retries:
                    log.error("Max retries exhausted")
                    return
                continue

            # Step 6: write files to workspace
            artifact_dir = self.artifacts_root / f"gen-{gen_number}"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            for file_path, content in parsed_files.items():
                dest = artifact_dir / file_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)

            # Copy spec file
            spec_dest_dir = artifact_dir / "spec"
            spec_dest_dir.mkdir(parents=True, exist_ok=True)
            spec_dest = spec_dest_dir / self.spec_path.name
            spec_dest.write_bytes(self.spec_path.read_bytes())

            # Build file list
            all_files: list[str] = []
            for p in artifact_dir.rglob("*"):
                if p.is_file() and p.name != "manifest.json":
                    rel = str(p.relative_to(artifact_dir))
                    all_files.append(rel)
            all_files.append("manifest.json")

            # Extract contracts from spec
            contracts = extract_contracts_from_spec(spec_content)

            # Build and write manifest
            manifest = build_manifest(
                generation=gen_number,
                parent_generation=parent_gen,
                spec_hash=spec_hash,
                artifact_root=artifact_dir,
                files=all_files,
                producer_model=model,
                token_input=token_input,
                token_output=token_output,
                contracts=contracts,
            )
            write_manifest(artifact_dir, manifest)

            log.info(
                "Artifact written",
                generation=gen_number,
                artifact_dir=str(artifact_dir),
                file_count=len(all_files),
            )

            # Step 7: POST /spawn
            try:
                spawn_resp = await supervisor.spawn(
                    spec_hash=spec_hash,
                    generation=gen_number,
                    artifact_path=f"gen-{gen_number}",
                )
            except Exception as e:
                log.error("Failed to spawn", error=str(e))
                await asyncio.sleep(5)
                continue

            if not spawn_resp.get("ok"):
                log.error("Spawn failed", response=spawn_resp)
                retry_count += 1
                gens_attempted += 1
                continue

            log.info("Spawn successful", generation=gen_number)

            # Step 8: poll until outcome != in_progress
            gen_record: dict[str, Any] = {}
            while True:
                await asyncio.sleep(2)
                try:
                    versions = await supervisor.get_versions()
                except Exception as e:
                    log.warning("Poll failed", error=str(e))
                    continue

                found = next(
                    (r for r in versions if r.get("generation") == gen_number),
                    None,
                )
                if found is None:
                    log.warning("Generation record not found", generation=gen_number)
                    continue

                gen_record = found
                outcome = gen_record.get("outcome", "in_progress")
                if outcome != "in_progress":
                    break

            # Step 9: decide
            viability = gen_record.get("viability", {}) if gen_record else {}
            status = viability.get("status", "non-viable") if viability else "non-viable"

            gens_attempted += 1

            if status == "viable":
                log.info("Generation viable, promoting", generation=gen_number)
                try:
                    await supervisor.promote(gen_number)
                except Exception as e:
                    log.error("Promote failed", error=str(e))
                retry_count = 0
                log.info("Generation loop complete — viable generation produced")
                return
            else:
                log.warning("Generation non-viable, rolling back", generation=gen_number)
                try:
                    await supervisor.rollback(gen_number)
                except Exception as e:
                    log.error("Rollback failed", error=str(e))

                retry_count += 1
                if retry_count > self.max_retries:
                    log.error("Max retries exhausted, stopping")
                    return

                log.info(
                    "Will retry",
                    retry_count=retry_count,
                    max_retries=self.max_retries,
                )

        log.info("Max generations reached, stopping", max_gens=self.max_gens)