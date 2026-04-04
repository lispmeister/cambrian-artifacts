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


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def select_model(retry_count: int) -> str:
    """Select LLM model based on retry count."""
    if retry_count == 0:
        return DEFAULT_MODEL
    return ESCALATION_MODEL


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
- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use\
 triple quotes (\"\"\" or ''') for multi-line strings. Use \\n for embedded newlines in\
 single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.
- Test strings that embed XML-like content MUST use raw strings (r\"...\") or\
 triple-quoted strings to avoid escaping issues.
"""


def build_fresh_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> dict[str, Any]:
    """Build the fresh generation prompt."""
    user_message = f"""# Specification

{spec_content}

# Generation History

{json.dumps(history, indent=2)}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""
    return {
        "system": SYSTEM_PROMPT,
        "user": user_message,
    }


def build_informed_retry_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_gen: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> dict[str, Any]:
    """Build the informed retry prompt with failure context."""
    # Build failed code section
    failed_code_sections: list[str] = []
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = ext if ext in ("py", "txt", "md", "json", "toml", "ini", "cfg") else ""
        failed_code_sections.append(f"### {file_path}\n```{lang}\n{content}\n```")

    failed_code_str = "\n\n".join(failed_code_sections) if failed_code_sections else "(no files recovered)"

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    user_message = f"""# Specification

{spec_content}

# Generation History

{json.dumps(history, indent=2)}

# Previous Attempt Failed

Generation {failed_gen} failed at stage: {stage}
Summary: {summary}

## Failed Source Code

{failed_code_str}

## Diagnostics

{json.dumps(diagnostics, indent=2)}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {generation}
Parent generation: {parent}
"""
    return {
        "system": SYSTEM_PROMPT,
        "user": user_message,
    }


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> dict[str, Any]:
    """Build the parse repair prompt."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return {
        "system": SYSTEM_PROMPT,
        "user": user_message,
    }


async def call_llm(prompt: dict[str, Any], model: str) -> str:
    """Call the Anthropic LLM API and return the raw text response."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("calling_llm", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=prompt["system"],
        messages=[
            {"role": "user", "content": prompt["user"]},
        ],
    ) as stream:
        message = await stream.get_final_message()

    text_content = ""
    for block in message.content:
        if hasattr(block, "text"):
            text_content += block.text

    log.info(
        "llm_response_received",
        component="prime",
        model=model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )

    return text_content


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle embedded file tags.

    Critical behavior:
    - A <file path="..."> tag inside an existing block does NOT open a new block.
    - The close tag </file:end> must match the ENTIRE stripped line.
    - A line like 'x = "</file:end>"' does NOT close the block.
    """
    current_path: str | None = None
    current_lines: list[str] = []
    files: dict[str, str] = {}

    for line in response.splitlines(keepends=True):
        if current_path is None:
            # Looking for an opening tag
            m = re.match(r'^<file path="([^"]+)">', line)
            if m:
                current_path = m.group(1)
                current_lines = []
        else:
            # Inside a block: only exact match closes it
            if line.rstrip("\n\r") == "</file:end>":
                files[current_path] = "".join(current_lines)
                current_path = None
            else:
                # Accumulate content — including any <file> tags or </file:end> substrings
                current_lines.append(line)

    if current_path is not None:
        raise ParseError(f"Unclosed <file path={current_path!r}> block")

    return files
