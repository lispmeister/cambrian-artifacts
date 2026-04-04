"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

DEFAULT_MODEL = "claude-sonnet-4-6"
ESCALATION_MODEL = "claude-opus-4-6"

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
    """Raised when LLM response cannot be parsed into files."""


def get_model(retry_count: int) -> str:
    """Return the model to use based on retry count."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", DEFAULT_MODEL)
    return os.environ.get("CAMBRIAN_ESCALATION_MODEL", ESCALATION_MODEL)


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


def build_fresh_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
) -> dict[str, str]:
    """Build the user message for a fresh generation."""
    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""
    return {"system": SYSTEM_PROMPT, "user": user_message}


def build_informed_retry_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> dict[str, str]:
    """Build the user message for an informed retry."""
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    files_section_parts = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        files_section_parts.append(f"### {file_path}\n```{ext}\n{content}\n```")
    files_section = "\n\n".join(files_section_parts)

    import json
    diagnostics_json = json.dumps(diagnostics, indent=2)

    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_generation} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{files_section}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}
"""
    return {"system": SYSTEM_PROMPT, "user": user_message}


def build_parse_repair_prompt(parse_error_message: str, raw_response: str) -> dict[str, str]:
    """Build the user message for a parse repair attempt."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return {"system": SYSTEM_PROMPT, "user": user_message}


async def call_llm(model: str, prompt: dict[str, str]) -> str:
    """Call the Anthropic LLM and return the response text."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("llm_call", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=prompt["system"],
        messages=[{"role": "user", "content": prompt["user"]}],
    ) as stream:
        message = await stream.get_final_message()

    content = message.content[0]
    if hasattr(content, "text"):
        return content.text
    return str(content)
