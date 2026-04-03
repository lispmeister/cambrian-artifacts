"""LLM integration — prompt building, API calls, response parsing."""
from __future__ import annotations

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

# Module-level token usage storage (updated after each LLM call)
_last_token_usage: dict[str, int] = {"input": 0, "output": 0}


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine — NOT a regex — to handle embedded tags.

    The close tag </file:end> is recognized ONLY when it appears alone on a line
    (after stripping newline characters). A line like 'x = "</file:end>"' does
    NOT close the block.
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


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> tuple[str, str]:
    """Build the fresh generation prompt (system, user)."""
    history_json = json.dumps(generation_records, indent=2)

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


def build_informed_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> tuple[str, str]:
    """Build the informed retry prompt (system, user)."""
    history_json = json.dumps(generation_records, indent=2)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    # Build failed source code section
    failed_code_sections: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        failed_code_sections.append(
            f"### {file_path}\n```{ext}\n{content}\n```"
        )
    failed_code_str = "\n\n".join(failed_code_sections)

    user_msg = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_generation} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_str}

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
    parse_error_message: str,
    raw_response: str,
) -> tuple[str, str]:
    """Build the parse repair prompt (system, user)."""
    user_msg = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return SYSTEM_PROMPT, user_msg


def get_model(base_model: str, escalation_model: str, retry_count: int) -> str:
    """Select model based on retry count."""
    if retry_count >= 1:
        return escalation_model
    return base_model


async def call_llm(model: str, system: str, user: str) -> str:
    """Call the Anthropic LLM API using streaming."""
    global _last_token_usage
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    log.info("llm_call_start", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = await stream.get_final_message()

    _last_token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info("llm_call_complete", component="prime", model=model,
             input_tokens=_last_token_usage["input"],
             output_tokens=_last_token_usage["output"])

    # Extract text content
    text_parts: list[str] = []
    for block in message.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    return "".join(text_parts)
