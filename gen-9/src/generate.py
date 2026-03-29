#!/usr/bin/env python3
"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import structlog

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
    """Raised when LLM response cannot be parsed into files."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The state machine stops at the FIRST line that is exactly '</file:end>'
    (no other content on that line). Content lines that merely contain
    '</file:end>' as a substring (not the entire line) are kept as content.
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


def build_fresh_prompt(
    spec_content: str,
    generation: int,
    parent: int,
    history: list[dict[str, Any]],
) -> str:
    """Build the user message for a fresh generation attempt."""
    history_json = json.dumps(history, indent=2)
    return (
        f"# Specification\n\n"
        f"{spec_content}\n\n"
        f"# Generation History\n\n"
        f"{history_json}\n\n"
        f"# Task\n\n"
        f"Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_retry_prompt(
    spec_content: str,
    generation: int,
    parent: int,
    history: list[dict[str, Any]],
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    """Build the user message for an informed retry after failure."""
    history_json = json.dumps(history, indent=2)
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "")
    diagnostics_json = json.dumps(diagnostics, indent=2)

    file_sections = []
    for file_path, content in sorted(failed_files.items()):
        ext = Path(file_path).suffix.lstrip(".")
        if not ext:
            ext = "text"
        file_sections.append(
            f"### {file_path}\n```{ext}\n{content}\n```"
        )
    files_text = "\n\n".join(file_sections)

    return (
        f"# Specification\n\n"
        f"{spec_content}\n\n"
        f"# Generation History\n\n"
        f"{history_json}\n\n"
        f"# Previous Attempt Failed\n\n"
        f"Generation {failed_generation} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        f"## Failed Source Code\n\n"
        f"{files_text}\n\n"
        f"## Diagnostics\n\n"
        f"{diagnostics_json}\n\n"
        f"# Task\n\n"
        f"The previous attempt failed. Study the failed code and diagnostics above.\n"
        f"Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> str:
    """Build the user message for a parse repair attempt."""
    return (
        f"# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        f"# Malformed Response\n\n"
        f"{raw_response}\n\n"
        f"# Task\n\n"
        "Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        "matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )


def select_model(retry_count: int) -> str:
    """Select LLM model based on retry count."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


@dataclass
class GenerationConfig:
    """Configuration for the generation loop."""
    spec_path: Path
    supervisor_url: str
    model: str
    escalation_model: str
    max_gens: int
    max_retries: int
    max_parse_retries: int
    workspace: Path
    artifacts_root: Path
    token_budget: int

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        return cls(
            spec_path=Path(os.environ.get(
                "CAMBRIAN_SPEC_PATH", "./spec/CAMBRIAN-SPEC-005.md"
            )),
            supervisor_url=os.environ.get(
                "CAMBRIAN_SUPERVISOR_URL", "http://host.docker.internal:8400"
            ),
            model=os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6"),
            escalation_model=os.environ.get(
                "CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6"
            ),
            max_gens=int(os.environ.get("CAMBRIAN_MAX_GENS", "5")),
            max_retries=int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3")),
            max_parse_retries=int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2")),
            workspace=Path(os.environ.get("CAMBRIAN_WORKSPACE", "/workspace")),
            artifacts_root=Path(os.environ.get("CAMBRIAN_ARTIFACTS_ROOT", "/workspace")),
            token_budget=int(os.environ.get("CAMBRIAN_TOKEN_BUDGET", "0")),
        )


async def call_llm(
    user_message: str,
    model: str,
    system: str = SYSTEM_PROMPT,
) -> tuple[str, dict[str, int]]:
    """
    Call the Anthropic LLM API using streaming.

    Returns (response_text, token_usage).
    token_usage has keys 'input' and 'output'.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )

    async with client.messages.stream(
        model=model,
        max_tokens=16384,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    response_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            response_text += block.text

    token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    return response_text, token_usage


async def call_llm_with_parse_repair(
    user_message: str,
    model: str,
    max_parse_retries: int,
    system: str = SYSTEM_PROMPT,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Call the LLM and attempt parse repair if needed.

    Returns (files_dict, combined_token_usage).
    Raises ParseError if all repair attempts fail.
    """
    response, usage = await call_llm(user_message, model, system)
    total_input = usage["input"]
    total_output = usage["output"]

    for attempt in range(max_parse_retries + 1):
        try:
            files = parse_files(response)
            return files, {"input": total_input, "output": total_output}
        except ParseError as e:
            if attempt >= max_parse_retries:
                raise
            logger.warning(
                "parse error, attempting repair",
                attempt=attempt,
                error=str(e),
            )
            repair_prompt = build_parse_repair_prompt(str(e), response)
            response, repair_usage = await call_llm(repair_prompt, model, system)
            total_input += repair_usage["input"]
            total_output += repair_usage["output"]

    # Should not reach here
    raise ParseError("All parse repair attempts failed")


async def run_loop(
    config: GenerationConfig,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Main generation loop."""
    from src.supervisor import SupervisorClient
    from src.manifest import build_manifest, write_manifest, compute_spec_hash, extract_contracts_from_spec

    def set_status(s: str) -> None:
        if status_callback:
            status_callback(s)

    supervisor = SupervisorClient(config.supervisor_url)

    # Read spec
    if not config.spec_path.exists():
        logger.error("spec file not found", path=str(config.spec_path))
        return

    spec_content = config.spec_path.read_text(encoding="utf-8")
    spec_hash = compute_spec_hash(config.spec_path)
    logger.info("spec loaded", spec_hash=spec_hash)

    retry_count = 0
    failed_generation: int | None = None
    failed_diagnostics: dict[str, Any] | None = None
    failed_files: dict[str, str] = {}
    gen_attempts = 0

    while gen_attempts < config.max_gens:
        gen_attempts += 1

        # Step 1: Determine generation number
        try:
            history = await supervisor.get_versions()
        except Exception as exc:
            logger.error("failed to get versions from supervisor", error=str(exc))
            await asyncio.sleep(5)
            continue

        if history:
            next_gen = max(r.get("generation", 0) for r in history) + 1
        else:
            next_gen = 1
        parent_gen = next_gen - 1

        logger.info(
            "starting generation",
            generation=next_gen,
            retry_count=retry_count,
        )

        # Step 3: Build prompt
        set_status("generating")
        model = select_model(retry_count)

        if retry_count > 0 and failed_diagnostics is not None and failed_generation is not None:
            user_message = build_retry_prompt(
                spec_content=spec_content,
                generation=next_gen,
                parent=parent_gen,
                history=history,
                failed_generation=failed_generation,
                diagnostics=failed_diagnostics,
                failed_files=failed_files,
            )
        else:
            user_message = build_fresh_prompt(
                spec_content=spec_content,
                generation=next_gen,
                parent=parent_gen,
                history=history,
            )

        # Step 4 & 5: Call LLM and parse response
        try:
            files, token_usage = await call_llm_with_parse_repair(
                user_message=user_message,
                model=model,
                max_parse_retries=config.max_parse_retries,
            )
        except ParseError as exc:
            logger.error("parse failed after all repairs", error=str(exc))
            retry_count += 1
            if retry_count > config.max_retries:
                logger.error("max retries exceeded")
                return
            continue
        except Exception as exc:
            logger.error("LLM call failed", error=str(exc))
            retry_count += 1
            if retry_count > config.max_retries:
                return
            continue

        # Step 6: Write files to workspace
        artifact_dir = config.artifacts_root / f"gen-{next_gen}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            dest = artifact_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Copy spec file
        spec_dest_dir = artifact_dir / "spec"
        spec_dest_dir.mkdir(parents=True, exist_ok=True)
        spec_filename = config.spec_path.name
        spec_dest = spec_dest_dir / spec_filename
        spec_dest.write_bytes(config.spec_path.read_bytes())

        # Extract contracts from spec
        contracts = extract_contracts_from_spec(spec_content)

        # Build and write manifest
        all_file_paths = list(files.keys()) + [f"spec/{spec_filename}"]
        manifest = build_manifest(
            generation=next_gen,
            parent_generation=parent_gen,
            spec_hash=spec_hash,
            artifact_root=artifact_dir,
            files=all_file_paths,
            model=model,
            token_usage=token_usage,
            contracts=contracts,
        )
        write_manifest(artifact_dir, manifest)

        logger.info(
            "artifact written",
            generation=next_gen,
            files=len(files),
            artifact_dir=str(artifact_dir),
        )

        # Step 7: Request verification
        set_status("verifying")
        try:
            spawn_result = await supervisor.spawn(
                spec_hash=spec_hash,
                generation=next_gen,
                artifact_path=f"gen-{next_gen}",
            )
            if not spawn_result.get("ok"):
                logger.error("spawn failed", result=spawn_result)
                retry_count += 1
                if retry_count > config.max_retries:
                    return
                continue
        except Exception as exc:
            logger.error("spawn request failed", error=str(exc))
            retry_count += 1
            if retry_count > config.max_retries:
                return
            continue

        # Step 8: Poll for completion
        generation_record = None
        while True:
            await asyncio.sleep(2)
            try:
                versions = await supervisor.get_versions()
                for record in versions:
                    if record.get("generation") == next_gen:
                        generation_record = record
                        break
                if generation_record and generation_record.get("outcome") != "in_progress":
                    break
            except Exception as exc:
                logger.warning("poll failed", error=str(exc))

        # Step 9: Decide
        viability = generation_record.get("viability") if generation_record else None
        outcome = generation_record.get("outcome") if generation_record else "failed"

        if viability and viability.get("status") == "viable":
            # Promote
            try:
                await supervisor.promote(next_gen)
                logger.info("generation promoted", generation=next_gen)
                retry_count = 0
                failed_generation = None
                failed_diagnostics = None
                failed_files = {}
                set_status("idle")
                return
            except Exception as exc:
                logger.error("promote failed", error=str(exc))
                return
        else:
            # Rollback
            try:
                await supervisor.rollback(next_gen)
            except Exception as exc:
                logger.error("rollback failed", error=str(exc))

            retry_count += 1
            if retry_count > config.max_retries:
                logger.error("max retries exceeded, stopping")
                set_status("idle")
                return

            # Collect failure context for informed retry
            failed_generation = next_gen
            if viability:
                failed_diagnostics = viability.get("diagnostics", {})
            else:
                failed_diagnostics = {}

            # Read failed files from disk
            failed_files = {}
            for rel_path in files.keys():
                file_path = artifact_dir / rel_path
                if file_path.exists():
                    try:
                        failed_files[rel_path] = file_path.read_text(encoding="utf-8")
                    except Exception:
                        pass

            logger.warning(
                "generation failed, will retry",
                generation=next_gen,
                retry_count=retry_count,
            )

    logger.error("max generations reached")
    set_status("idle")