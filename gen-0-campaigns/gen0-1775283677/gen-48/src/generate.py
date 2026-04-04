"""LLM integration: prompt building, API calls, response parsing."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

_last_token_usage: dict[str, int] | None = None


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into files."""
    pass


def get_model(retry_count: int) -> str:
    """Return the model to use based on retry count."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The close-tag match is EXACT: the entire line (stripped of newlines) must be
    exactly </file:end>. A line containing </file:end> as a substring does NOT close.
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


def _get_system_prompt() -> str:
    """Return the system prompt for code generation."""
    return (
        "You are a code generator. You produce complete, working Python codebases from specifications.\n"
        "\n"
        "Rules:\n"
        '- Output ONLY <file path="...">content</file:end> blocks. One block per file.\n'
        "- Every file needed to build, test, and run the project must be in a <file> block.\n"
        "- Include a requirements.txt with all dependencies.\n"
        "- Include a test suite that exercises all functionality.\n"
        "- The code must work in Python 3.14 inside a Docker container with a venv at /venv.\n"
        "- Do NOT include manifest.json — it is generated separately.\n"
        "- Do NOT include the spec file — it is copied separately.\n"
        "- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use\n"
        '  triple quotes (""" or \'\'\') for multi-line strings. Use \\n for embedded newlines in\n'
        "  single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.\n"
        "- Test strings that embed XML-like content MUST use raw strings (r\"...\") or\n"
        "  triple-quoted strings to avoid escaping issues.\n"
    )


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
) -> tuple[str, str]:
    """Build the fresh generation prompt."""
    system_msg = _get_system_prompt()

    history_json = json.dumps(generation_records, indent=2)

    user_msg = (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{history_json}\n\n"
        "# Task\n\n"
        "Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {offspring_gen}\n"
        f"Parent generation: {parent_gen}\n"
    )

    return system_msg, user_msg


def build_informed_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
    failed_files: dict[str, str],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Build the informed retry prompt with failure context."""
    system_msg = _get_system_prompt()

    history_json = json.dumps(generation_records, indent=2)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    prev_gen = offspring_gen - 1
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    failed_code_sections = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        failed_code_sections.append(
            f"### {file_path}\n```{ext}\n{content}\n```"
        )
    failed_code_str = "\n\n".join(failed_code_sections)

    user_msg = (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{history_json}\n\n"
        "# Previous Attempt Failed\n\n"
        f"Generation {prev_gen} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        "## Failed Source Code\n\n"
        f"{failed_code_str}\n\n"
        "## Diagnostics\n\n"
        f"{diagnostics_json}\n\n"
        "# Task\n\n"
        "The previous attempt failed. Study the failed code and diagnostics above.\n"
        "Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {offspring_gen}\n"
        f"Parent generation: {parent_gen}\n"
    )

    return system_msg, user_msg


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> tuple[str, str]:
    """Build the parse repair prompt."""
    system_msg = _get_system_prompt()

    user_msg = (
        "# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        "# Malformed Response\n\n"
        f"{raw_response}\n\n"
        "# Task\n\n"
        "Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        "matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )

    return system_msg, user_msg


async def call_llm(
    system_msg: str,
    user_msg: str,
    model: str,
) -> str:
    """Call the Anthropic LLM API using streaming."""
    global _last_token_usage

    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info(
        "llm_call_starting",
        component="prime",
        model=model,
    )

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        message = await stream.get_final_message()

    _last_token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    # Extract text content
    text_parts = []
    for block in message.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    result = "".join(text_parts)

    log.info(
        "llm_call_complete",
        component="prime",
        model=model,
        input_tokens=_last_token_usage["input"],
        output_tokens=_last_token_usage["output"],
    )

    return result
