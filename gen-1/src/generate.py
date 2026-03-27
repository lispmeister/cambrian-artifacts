"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import json
import re
from typing import Any

import anthropic
import structlog

log = structlog.get_logger(component="prime")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

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
  triple-quoted strings for multi-line content. Use \\n for embedded newlines in
  single-line strings. A bare newline inside "..." or '...' is a SyntaxError.
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings
  (r"...") or triple-quoted strings to avoid escaping issues.
"""


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> str:
    """Build the user message for a fresh generation attempt."""
    records_json = json.dumps(generation_records, indent=2) if generation_records else "[]"
    return f"""\
# Specification

{spec_content}

# Generation History

{records_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""


def build_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    """Build the user message for an informed retry after failure."""
    records_json = json.dumps(generation_records, indent=2) if generation_records else "[]"
    diag_json = json.dumps(diagnostics, indent=2)

    # Build failed source code section
    source_sections = []
    for path, content in failed_files.items():
        lang = "python" if path.endswith(".py") else ""
        source_sections.append(f"### {path}\n```{lang}\n{content}\n```")
    source_text = "\n\n".join(source_sections) if source_sections else "(no source code available)"

    stage = diagnostics.get("stage", diagnostics.get("failure_stage", "unknown"))
    summary = diagnostics.get("summary", diagnostics.get("error", "unknown error"))

    return f"""\
# Specification

{spec_content}

# Generation History

{records_json}

# Previous Attempt Failed

Generation {failed_generation} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{source_text}

## Diagnostics

{diag_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}
"""


def build_parse_repair_prompt(parse_error: str, raw_response: str) -> str:
    """Build the user message for a parse repair attempt after ParseError."""
    return f"""\
# Parse Error

The previous response could not be parsed. Error: {parse_error}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def call_llm(system: str, user: str, model: str) -> tuple[str, int, int]:
    """Call the Anthropic API. Returns (response_text, input_tokens, output_tokens).

    Uses streaming to avoid the SDK's 10-minute non-streaming timeout, which is
    triggered when max_tokens is large enough that the request could exceed the limit.
    """
    client = anthropic.AsyncAnthropic()
    log.info("llm_call_start", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=32768,
        system=system,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = await stream.get_final_message()

    # Validate response has text content
    if not message.content:
        raise RuntimeError("LLM returned empty content")
    text_blocks = [b for b in message.content if b.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"LLM returned no text blocks, got types: {[b.type for b in message.content]}")

    text = text_blocks[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    log.info("llm_call_complete", input_tokens=input_tokens, output_tokens=output_tokens)
    return text, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class ParseError(Exception):
    """Raised when the LLM response cannot be parsed into file blocks."""


def parse_files(response: str) -> dict[str, str]:
    """Extract files from <file path="...">content</file:end> blocks.

    Uses a line-by-line state machine instead of a dotall regex. Immune to the
    nested-tag truncation that a dotall regex produces when file content contains
    <file> or </file> literals (e.g. in test fixtures).

    Raises ParseError if any <file> block is opened without a closing </file:end>.
    Anything outside <file> blocks is silently discarded.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    for line in response.splitlines(keepends=True):
        if current_path is None:
            m = re.match(r'<file path="([^"]+)">(.*)', line)
            if m:
                current_path = m.group(1)
                # Strip trailing CRLF/LF from the opening-tag line's remainder.
                # re.(.*)  stops before \n but may include a lone \r from CRLF input.
                rest = m.group(2).rstrip("\r\n")
                current_lines = [rest + "\n"] if rest else []
        elif line.rstrip("\n\r") == "</file:end>":
            content = "".join(current_lines).replace("\r\n", "\n").strip("\n")
            files[current_path] = content
            current_path = None
        else:
            current_lines.append(line)

    if current_path is not None:
        raise ParseError(f"Unclosed <file path={current_path!r}> block")

    return files
