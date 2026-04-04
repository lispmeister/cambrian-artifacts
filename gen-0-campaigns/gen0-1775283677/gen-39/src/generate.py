"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

# Track last token usage globally
_last_token_usage: dict[str, int] = {"input": 0, "output": 0}

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
- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use triple
  quotes (\"\"\" or ''') for multi-line strings. Use \\n for embedded newlines in
  single-line strings. A bare newline inside "..." or '...' is a SyntaxError.
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings
  (r"...") or triple-quoted strings to avoid escaping issues.
"""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The close-tag match is EXACT: the entire line (stripped of newlines) must be
    exactly </file:end>. A line containing </file:end> as a substring does NOT match.
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
    history_json: str,
    generation: int,
    parent: int,
) -> str:
    """Build the user message for a fresh generation attempt."""
    return (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{history_json}\n\n"
        "# Task\n\n"
        "Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_retry_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
    failed_gen: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    """Build the user message for an informed retry after failure."""
    import json

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "")
    diagnostics_json = json.dumps(diagnostics, indent=2)

    # Build failed source code section
    file_sections = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        file_sections.append(f"### {file_path}\n```{ext}\n{content}\n```")
    failed_code_section = "\n\n".join(file_sections)

    return (
        "# Specification\n\n"
        f"{spec_content}\n\n"
        "# Generation History\n\n"
        f"{history_json}\n\n"
        "# Previous Attempt Failed\n\n"
        f"Generation {failed_gen} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        "## Failed Source Code\n\n"
        f"{failed_code_section}\n\n"
        "## Diagnostics\n\n"
        f"{diagnostics_json}\n\n"
        "# Task\n\n"
        "The previous attempt failed. Study the failed code and diagnostics above.\n"
        "Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_parse_repair_prompt(parse_error_message: str, raw_response: str) -> str:
    """Build the user message for a parse repair attempt."""
    return (
        "# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        "# Malformed Response\n\n"
        f"{raw_response}\n\n"
        "# Task\n\n"
        "Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        "matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )


def select_model(retry_count: int) -> str:
    """Select model based on retry count."""
    if retry_count >= 1:
        return ESCALATION_MODEL
    return MODEL


async def call_llm(system_prompt: str, user_message: str, model: str) -> str:
    """Call the Anthropic LLM API using streaming."""
    global _last_token_usage

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    log.info("calling_llm", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    # Extract token usage
    if message.usage:
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
    log.info("llm_response_received", component="prime", model=model,
             input_tokens=_last_token_usage["input"],
             output_tokens=_last_token_usage["output"])
    return result
