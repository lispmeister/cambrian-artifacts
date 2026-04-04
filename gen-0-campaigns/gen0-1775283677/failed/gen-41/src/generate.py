"""LLM integration: prompt building, API calls, response parsing."""

import json
import os
import re
from typing import Any

import structlog

log = structlog.get_logger()


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.

    Uses a line-by-line state machine. The close tag must be the ENTIRE line
    (stripped of newlines). A line like 'x = "</file:end>"' does NOT close a block
    because the stripped line is NOT equal to '</file:end>'.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    for line in response.splitlines(keepends=True):
        if current_path is None:
            m = re.match(r'^<file path="([^"]+)">', line)
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
- entry.start in manifest MUST use module form: python -m src.prime (NOT python src/prime.py).
- src/__init__.py MUST exist (may be empty).
- structlog: first positional arg IS the event. log.info("event_name", key=val). NEVER log.info("event", event="x").
- get_model() MUST read os.environ at call time, not at module import time.
"""


def build_fresh_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> list[dict[str, str]]:
    """Build the initial generation prompt."""
    history_json = json.dumps(history, indent=2)
    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""
    return [{"role": "user", "content": user_message}]


def build_retry_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> list[dict[str, str]]:
    """Build the informed retry prompt with failure context."""
    history_json = json.dumps(history, indent=2)
    diagnostics_json = json.dumps(diagnostics, indent=2)

    failed_stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    files_section_parts = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        files_section_parts.append(f"### {file_path}\n```{ext}\n{content}\n```")
    files_section = "\n\n".join(files_section_parts)

    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {generation - 1} failed at stage: {failed_stage}
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
    return [{"role": "user", "content": user_message}]


def build_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> list[dict[str, str]]:
    """Build a parse repair prompt."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return [{"role": "user", "content": user_message}]


def get_model(retry_count: int) -> str:
    """Get the model to use based on retry count. Reads env at call time."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    else:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 32000,
) -> str:
    """Call the Anthropic LLM API using streaming."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    log.info("llm_call_starting", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,  # type: ignore[arg-type]
    ) as stream:
        message = await stream.get_final_message()

    content = message.content[0]
    if hasattr(content, "text"):
        text = content.text
    else:
        text = str(content)

    input_tokens = message.usage.input_tokens if message.usage else 0
    output_tokens = message.usage.output_tokens if message.usage else 0

    log.info(
        "llm_call_complete",
        component="prime",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return text
