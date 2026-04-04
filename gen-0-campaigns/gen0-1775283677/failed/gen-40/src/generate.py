"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

CAMBRIAN_MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
CAMBRIAN_ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

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
  triple-quoted strings to avoid escaping issues.
"""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""


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


def get_model(retry_count: int) -> str:
    """Select model based on retry count."""
    if retry_count == 0:
        return CAMBRIAN_MODEL
    return CAMBRIAN_ESCALATION_MODEL


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
) -> list[dict[str, str]]:
    """Build the user message for a fresh generation attempt."""
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
    return [{"role": "user", "content": user_message}]


def build_informed_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
    failed_context: dict[str, Any],
) -> list[dict[str, str]]:
    """Build the user message for an informed retry after failure."""
    history_json = json.dumps(generation_records, indent=2)
    diagnostics = failed_context.get("diagnostics", {})
    failed_files = failed_context.get("failed_files", {})
    failed_gen = failed_context.get("failed_generation", offspring_gen - 1)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    failed_code_sections: list[str] = []
    for file_path, content in failed_files.items():
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        failed_code_sections.append(f"### {file_path}\n```{ext}\n{content}\n```")
    failed_code_str = "\n\n".join(failed_code_sections)

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_gen} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_str}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {offspring_gen}
Parent generation: {parent_gen}
"""
    return [{"role": "user", "content": user_message}]


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> list[dict[str, str]]:
    """Build the user message for a parse repair attempt."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return [{"role": "user", "content": user_message}]


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 32768,
) -> str:
    """Call the Anthropic LLM API using streaming."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info(
        "llm_call_starting",
        component="prime",
        model=model,
        max_tokens=max_tokens,
    )

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,  # type: ignore[arg-type]
    ) as stream:
        message = await stream.get_final_message()

    content = message.content
    if not content:
        return ""

    text_parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    result = "".join(text_parts)

    log.info(
        "llm_call_complete",
        component="prime",
        model=model,
        input_tokens=message.usage.input_tokens if message.usage else 0,
        output_tokens=message.usage.output_tokens if message.usage else 0,
    )

    return result
