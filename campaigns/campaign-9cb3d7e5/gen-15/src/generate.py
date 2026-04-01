#!/usr/bin/env python3
"""LLM integration: prompt building, API calls, response parsing."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import anthropic
import structlog

logger = structlog.get_logger()

# Configuration
CAMBRIAN_MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
CAMBRIAN_ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
CAMBRIAN_MAX_GENS = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
CAMBRIAN_MAX_RETRIES = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
CAMBRIAN_MAX_PARSE_RETRIES = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))

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
  triple quotes (\"\"\" or ''') for multi-line strings. Use \\n for embedded newlines in
  single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.
- Test strings that embed XML-like content MUST use raw strings (r\"...\") or
  triple-quoted strings to avoid escaping issues.\
"""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The state machine stops at the FIRST </file:end> on its own line.
    Any text that looks like </file:end> inside content will terminate the block.
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


async def parse_files_with_repair(
    raw_response: str,
    model: str,
) -> dict[str, str]:
    """
    Attempt to parse files, retrying with repair prompts on ParseError.
    Raises ParseError if all repair attempts fail.
    """
    log = logger.bind(component="prime")
    current_response = raw_response

    for attempt in range(CAMBRIAN_MAX_PARSE_RETRIES + 1):
        try:
            return parse_files(current_response)
        except ParseError as e:
            if attempt >= CAMBRIAN_MAX_PARSE_RETRIES:
                log.error("All parse repair attempts exhausted", attempts=attempt + 1)
                raise

            log.warning(
                "Parse failed, attempting repair",
                attempt=attempt + 1,
                error=str(e),
            )
            repair_msg = build_parse_repair_prompt(str(e), current_response)
            repaired, _ = await call_llm(
                system_message=SYSTEM_PROMPT,
                user_message=repair_msg,
                model=model,
            )
            current_response = repaired

    raise ParseError("Parse repair loop exited unexpectedly")


def select_model(retry_count: int) -> str:
    """Select model based on retry count."""
    if retry_count == 0:
        return CAMBRIAN_MODEL
    return CAMBRIAN_ESCALATION_MODEL


def get_generation_number(versions: list[dict[str, Any]]) -> int:
    """Compute the next generation number from history."""
    if not versions:
        return 1
    max_gen = max(r.get("generation", 0) for r in versions)
    return max_gen + 1


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
) -> tuple[str, str]:
    """Build system and user messages for a fresh generation."""
    history_json = json.dumps(generation_records, indent=2)
    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {offspring_gen}
Parent generation: {parent_gen}
"""
    return SYSTEM_PROMPT, user_message


def build_informed_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> tuple[str, str]:
    """Build system and user messages for an informed retry."""
    history_json = json.dumps(generation_records, indent=2)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    # Build failed source code section
    failed_code_parts: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = ext if ext in ("py", "txt", "json", "md", "yaml", "yml", "toml", "ini") else ""
        failed_code_parts.append(f"### {file_path}\n```{lang}\n{content}\n```")

    failed_code_section = "\n\n".join(failed_code_parts)
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "")

    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {parent_gen} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_section}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {offspring_gen}
Parent generation: {parent_gen}
"""
    return SYSTEM_PROMPT, user_message


def build_parse_repair_prompt(parse_error_message: str, raw_response: str) -> str:
    """Build repair prompt for malformed LLM response."""
    return f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""


async def call_llm(
    system_message: str,
    user_message: str,
    model: str,
    max_tokens: int = 32768,
) -> tuple[str, dict[str, int]]:
    """Call the Anthropic LLM API using streaming."""
    log = logger.bind(component="prime")
    client = anthropic.AsyncAnthropic()

    log.info("Calling LLM", model=model, max_tokens=max_tokens)

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_message,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    content = ""
    for block in message.content:
        if hasattr(block, "text"):
            content += block.text

    token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info(
        "LLM call complete",
        model=model,
        input_tokens=token_usage["input"],
        output_tokens=token_usage["output"],
    )

    return content, token_usage
