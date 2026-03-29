#!/usr/bin/env python3
"""LLM integration: prompt building, API calls, response parsing."""

from __future__ import annotations

import os
import re
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="prime")

CAMBRIAN_MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
CAMBRIAN_ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
MAX_TOKENS = int(os.environ.get("CAMBRIAN_MAX_TOKENS", "32000"))

SYSTEM_PROMPT = (
    "You are a code generator. You produce complete, working Python codebases "
    "from specifications.\n\n"
    "Rules:\n"
    '- Output ONLY <file path="...">content</file:end> blocks. One block per file.\n'
    "- Every file needed to build, test, and run the project must be in a <file> block.\n"
    "- Include a requirements.txt with all dependencies.\n"
    "- Include a test suite that exercises all functionality.\n"
    "- The code must work in Python 3.14 inside a Docker container with a venv at /venv.\n"
    "- Do NOT include manifest.json — it is generated separately.\n"
    "- Do NOT include the spec file — it is copied separately.\n"
    "- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use\n"
    "  triple-quoted strings for multi-line content. Use \\n for embedded newlines in\n"
    "  single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.\n"
    "- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings\n"
    '  (r"...") or triple-quoted strings to avoid escaping issues.\n'
    "- LLM API: MUST use `async with client.messages.stream(...) as stream:` with\n"
    "  `await stream.get_final_message()`. Do NOT use `client.messages.create()` — the SDK\n"
    "  raises an error for large max_tokens with non-streaming calls.\n"
    "- aiohttp tests: use `aiohttp_client` pytest fixture. Do NOT use `AioHTTPTestCase` or\n"
    "  `@unittest_run_loop` — both are deprecated and break in aiohttp 3.8+.\n"
    '  Use `aiohttp_server` for mock servers. Set asyncio_mode = "auto" in pytest.ini.\n'
)


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The state machine stops at the FIRST line that is exactly '</file:end>'
    (after stripping trailing newline). A line containing '</file:end>' along
    with other text does NOT terminate the block.
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
    generation_records_json: str,
    generation: int,
    parent: int,
) -> tuple[str, str]:
    """Build system and user messages for a fresh (non-retry) generation."""
    user_msg = (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{generation_records_json}\n\n"
        "# Task\n\n"
        "Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )
    return SYSTEM_PROMPT, user_msg


def build_retry_prompt(
    spec_content: str,
    generation_records_json: str,
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> tuple[str, str]:
    """Build system and user messages for an informed retry."""
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown")
    diagnostics_json = _format_json(diagnostics)

    files_section_parts = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        files_section_parts.append(
            f"### {file_path}\n```{ext}\n{content}\n```"
        )
    files_section = "\n\n".join(files_section_parts)

    user_msg = (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{generation_records_json}\n\n"
        "# Previous Attempt Failed\n\n"
        f"Generation {failed_generation} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        "## Failed Source Code\n\n"
        f"{files_section}\n\n"
        "## Diagnostics\n\n"
        f"{diagnostics_json}\n\n"
        "# Task\n\n"
        "The previous attempt failed. Study the failed code and diagnostics above.\n"
        "Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )
    return SYSTEM_PROMPT, user_msg


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> tuple[str, str]:
    """Build system and user messages for a parse repair attempt."""
    user_msg = (
        "# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        "# Malformed Response\n\n"
        f"{raw_response}\n\n"
        "# Task\n\n"
        "Re-emit the EXACT SAME files using the correct format. Every <file> block MUST "
        "have a matching </file:end> on its own line. No nesting. No extra content "
        "between blocks.\n"
    )
    return SYSTEM_PROMPT, user_msg


async def call_llm(
    system_msg: str,
    user_msg: str,
    retry_count: int = 0,
) -> tuple[str, dict[str, int], str]:
    """
    Call the Anthropic LLM API using streaming.
    Returns (response_text, usage_dict, model_name).
    """
    import anthropic

    model = CAMBRIAN_MODEL if retry_count == 0 else CAMBRIAN_ESCALATION_MODEL
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    log = logger.bind(model=model, retry_count=retry_count)
    log.info("calling anthropic API")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        message = await stream.get_final_message()

    response_text = ""
    for block in message.content:
        if hasattr(block, "text"):
            response_text += block.text

    usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info("LLM call complete", input_tokens=usage["input"], output_tokens=usage["output"])
    return response_text, usage, model


def _format_json(data: dict[str, Any]) -> str:
    """Format dict as JSON string."""
    import json
    return json.dumps(data, indent=2)