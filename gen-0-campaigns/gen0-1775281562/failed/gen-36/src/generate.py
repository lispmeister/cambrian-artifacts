"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

DEFAULT_MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

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
    pass


def get_model(retry_count: int) -> str:
    """Return the appropriate model based on retry count."""
    if retry_count == 0:
        return DEFAULT_MODEL
    return ESCALATION_MODEL


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The close-tag match is exact: the ENTIRE line (stripped of newlines) must
    equal '</file:end>'. A line containing '</file:end>' as a substring but
    with other characters does NOT close the block.
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
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> tuple[str, str]:
    """Build system and user messages for a fresh generation attempt."""
    history_json = json.dumps(history, indent=2)
    user_msg = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""
    return SYSTEM_PROMPT, user_msg


def build_retry_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_gen: int,
    failed_files: dict[str, str],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Build system and user messages for an informed retry after failure."""
    history_json = json.dumps(history, indent=2)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    # Build failed source code section
    failed_code_parts: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = ext if ext else "text"
        failed_code_parts.append(f"### {file_path}\n```{lang}\n{content}\n```")
    failed_code_section = "\n\n".join(failed_code_parts)

    user_msg = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_gen} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_section}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}
"""
    return SYSTEM_PROMPT, user_msg


def build_parse_repair_prompt(
    parse_error: str,
    raw_response: str,
) -> str:
    """Build a user message to repair a malformed LLM response."""
    return f"""# Parse Error

The previous response could not be parsed. Error: {parse_error}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""


async def call_llm(
    system_msg: str,
    user_msg: str,
    model: str,
    max_tokens: int = 32000,
) -> tuple[str, dict[str, int]]:
    """
    Call the Anthropic LLM API using streaming.
    Returns (response_text, token_usage_dict).
    """
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("llm_call_start", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        message = await stream.get_final_message()

    text = ""
    for block in message.content:
        if hasattr(block, "text"):
            text += block.text

    token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info(
        "llm_call_done",
        component="prime",
        model=model,
        input_tokens=token_usage["input"],
        output_tokens=token_usage["output"],
    )

    return text, token_usage
