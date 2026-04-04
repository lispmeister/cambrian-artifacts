"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

# Stores token usage from the last LLM call
_last_token_usage: dict[str, int] = {"input": 0, "output": 0}

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_ESCALATION_MODEL = "claude-opus-4-6"

SYSTEM_PROMPT = """You are a code generator. You produce complete, working Python codebases from specifications.

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
- entry.start in manifest MUST use module form: python -m src.prime (not python src/prime.py)
- src/__init__.py MUST exist (may be empty)
- structlog: first positional arg IS the event key. log.info("event_name", key=val). NEVER log.info("event", event="name")
- get_model() and similar functions MUST read environment variables at call time, not at module import time"""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine. The close tag </file:end> must be
    the ENTIRE content of a line (after stripping newlines).
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
    """Return the appropriate model based on retry count. Reads env at call time."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", DEFAULT_MODEL)
    return os.environ.get("CAMBRIAN_ESCALATION_MODEL", DEFAULT_ESCALATION_MODEL)


def get_producer_model(retry_count: int) -> str:
    """Alias for get_model — returns the model used for production."""
    return get_model(retry_count)


def build_fresh_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
) -> dict[str, str]:
    """Build the prompt for a fresh (non-retry) generation."""
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


def build_retry_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> dict[str, str]:
    """Build the prompt for a retry after failure."""
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    failed_code_sections = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = ext if ext in ("py", "txt", "md", "json", "toml", "ini", "cfg") else ""
        failed_code_sections.append(f"### {file_path}\n```{lang}\n{content}\n```")

    failed_code_str = "\n\n".join(failed_code_sections)
    diagnostics_json = _safe_json(diagnostics)

    user_message = f"""# Specification

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
    return {"system": SYSTEM_PROMPT, "user": user_message}


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> dict[str, str]:
    """Build a prompt to repair a malformed LLM response."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return {"system": SYSTEM_PROMPT, "user": user_message}


def _safe_json(obj: Any) -> str:
    """Serialize object to JSON string, falling back to str() on error."""
    try:
        import json
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


async def call_llm(
    system_prompt: str,
    user_message: str,
    model: str,
) -> str:
    """Call the Anthropic API using streaming and return the response text."""
    global _last_token_usage

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    log.info(
        "llm_call_starting",
        component="prime",
        model=model,
    )

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    _last_token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info(
        "llm_call_complete",
        component="prime",
        model=model,
        input_tokens=_last_token_usage["input"],
        output_tokens=_last_token_usage["output"],
    )

    # Extract text from response
    text_parts = []
    for block in message.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    return "".join(text_parts)
