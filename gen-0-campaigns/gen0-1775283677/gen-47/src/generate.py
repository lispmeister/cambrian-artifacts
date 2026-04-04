"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

SYSTEM_PROMPT = """You are a code generator. You produce complete, working Python codebases from specifications.

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
  single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings
  (r\"...\") or triple-quoted strings to avoid escaping issues."""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine. NOT a regex.

    The close tag </file:end> MUST appear alone on its own line (after stripping newlines).
    A line containing </file:end> as part of longer content does NOT close the block.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    for line in response.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")
        if current_path is None:
            m = re.match(r'^<file path="([^"]+)">$', stripped)
            if m:
                current_path = m.group(1)
                current_lines = []
        elif stripped == "</file:end>":
            files[current_path] = "".join(current_lines)
            current_path = None
        else:
            current_lines.append(line)

    if current_path is not None:
        raise ParseError(f"Unclosed <file path={current_path!r}> block")

    return files


def get_model(retry_count: int) -> str:
    """Get the model to use based on retry count."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    else:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


def build_fresh_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
) -> tuple[str, str]:
    """Build a fresh generation prompt."""
    user_msg = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}"""
    return SYSTEM_PROMPT, user_msg


def build_retry_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
    failed_files: dict[str, str],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Build an informed retry prompt with failure context."""
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    failed_code_sections: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = ext if ext else "text"
        failed_code_sections.append(f"### {file_path}\n```{lang}\n{content}\n```")

    failed_code_str = "\n\n".join(failed_code_sections)
    import json
    diagnostics_json = json.dumps(diagnostics, indent=2)

    user_msg = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {generation - 1} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_str}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}"""
    return SYSTEM_PROMPT, user_msg


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_llm_response: str,
) -> tuple[str, str]:
    """Build a parse repair prompt."""
    user_msg = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_llm_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks."""
    return SYSTEM_PROMPT, user_msg


async def call_llm(system_msg: str, user_msg: str, model: str) -> str:
    """Call the LLM API and return the response text."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("llm_call_starting", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=system_msg,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        message = await stream.get_final_message()

    content = message.content[0]
    if hasattr(content, "text"):
        text = content.text
    else:
        text = str(content)

    input_tokens = message.usage.input_tokens if message.usage else 0
    output_tokens = message.usage.output_tokens if message.usage else 0

    log.info("llm_call_complete", component="prime", model=model,
             input_tokens=input_tokens, output_tokens=output_tokens)

    return text
