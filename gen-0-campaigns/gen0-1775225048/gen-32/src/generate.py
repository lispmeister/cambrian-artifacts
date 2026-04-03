"""LLM integration — prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any

import structlog

log = structlog.get_logger()

DEFAULT_MODEL = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
ESCALATION_MODEL = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")

# Last token usage from LLM call
_last_token_usage: dict[str, int] = {"input": 0, "output": 0}

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
  (r\"...\") or triple-quoted strings to avoid escaping issues.
- entry.start MUST use module form: python -m src.prime (NOT python src/prime.py)
- structlog: first positional arg IS the event. log.info("event_name", key=val). NEVER log.info("event", event="name")."""


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into files."""
    pass


def get_model(retry_count: int) -> str:
    """Return the model to use based on retry count."""
    if retry_count == 0:
        return DEFAULT_MODEL
    return ESCALATION_MODEL


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine. NOT a regex.

    The close-tag match is EXACT: the entire line (stripped of newlines) must equal
    '</file:end>'. A line containing '</file:end>' as a substring does NOT close the block.
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
    """Build the user message for a fresh generation."""
    return f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""


def build_retry_prompt(
    spec_content: str,
    history_json: str,
    generation: int,
    parent: int,
    failed_gen: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    """Build the user message for an informed retry."""
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")
    diag_json = _json_dumps(diagnostics)

    files_section_parts: list[str] = []
    for path, content in sorted(failed_files.items()):
        lang = _guess_language(path)
        files_section_parts.append(f"### {path}\n```{lang}\n{content}\n```")
    files_section = "\n\n".join(files_section_parts)

    return f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_gen} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{files_section}

## Diagnostics

{diag_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}
"""


def build_repair_prompt(parse_error_message: str, raw_response: str) -> str:
    """Build the user message for a parse repair attempt."""
    return f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""


async def call_llm(model: str, system_prompt: str, user_message: str) -> str:
    """Call the Anthropic LLM and return the response text."""
    global _last_token_usage

    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("llm_call_starting", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16384,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    _last_token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info("llm_call_complete", component="prime", model=model,
             input_tokens=_last_token_usage["input"],
             output_tokens=_last_token_usage["output"])

    return message.content[0].text


def _guess_language(path: str) -> str:
    """Guess language from file extension."""
    if path.endswith(".py"):
        return "python"
    if path.endswith(".md"):
        return "markdown"
    if path.endswith(".json"):
        return "json"
    if path.endswith(".toml"):
        return "toml"
    if path.endswith(".txt"):
        return "text"
    return ""


def _json_dumps(obj: Any) -> str:
    """JSON dump with indent."""
    import json
    return json.dumps(obj, indent=2)
