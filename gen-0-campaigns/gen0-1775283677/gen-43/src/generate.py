"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

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
    Uses a line-by-line state machine to handle nested/embedded file tags.

    The close tag </file:end> MUST be the entire line (stripped of newlines).
    A line containing </file:end> as part of a longer string does NOT close the block.
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
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    else:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


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
- entry.start in manifest MUST use module form: python -m src.prime (NOT python src/prime.py).
- src/__init__.py MUST exist (may be empty).
- All imports in test functions MUST be local to that function, not module-level."""


def build_fresh_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
) -> tuple[str, str]:
    """Build the fresh generation prompt."""
    history_json = json.dumps(generation_records, indent=2)
    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {offspring_gen}
Parent generation: {parent_gen}"""
    return SYSTEM_PROMPT, user_message


def build_informed_retry_prompt(
    spec_content: str,
    generation_records: list[dict[str, Any]],
    offspring_gen: int,
    parent_gen: int,
    failure_context: dict[str, Any],
) -> tuple[str, str]:
    """Build the informed retry prompt with failure context."""
    history_json = json.dumps(generation_records, indent=2)
    diagnostics = failure_context.get("diagnostics", {})
    failed_gen = failure_context.get("failed_generation", offspring_gen - 1)
    failed_files = failure_context.get("failed_files", {})
    diagnostics_json = json.dumps(diagnostics, indent=2)

    failed_code_sections: list[str] = []
    for file_path, content in failed_files.items():
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        failed_code_sections.append(f"### {file_path}\n```{ext}\n{content}\n```")

    failed_code_str = "\n\n".join(failed_code_sections)

    user_message = f"""# Specification

{spec_content}

# Generation History

{history_json}

# Previous Attempt Failed

Generation {failed_gen} failed at stage: {diagnostics.get('stage', 'unknown')}
Summary: {diagnostics.get('summary', 'No summary available')}

## Failed Source Code

{failed_code_str}

## Diagnostics

{diagnostics_json}

# Task

The previous attempt failed. Study the failed code and diagnostics above.
Produce a complete, corrected codebase that fixes the identified issues.
Generation number: {offspring_gen}
Parent generation: {parent_gen}"""
    return SYSTEM_PROMPT, user_message


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_response: str,
) -> tuple[str, str]:
    """Build the parse repair prompt."""
    user_message = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks."""
    return SYSTEM_PROMPT, user_message


async def call_llm(
    system_message: str,
    user_message: str,
    model: str,
) -> tuple[str, dict[str, int]]:
    """Call the Anthropic LLM API using streaming."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", "")
    )

    log.info("llm_call_starting", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=32000,
        system=system_message,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        message = await stream.get_final_message()

    content = ""
    for block in message.content:
        if hasattr(block, "text"):
            content += block.text

    token_usage = {
        "input": message.usage.input_tokens,
        "output": message.usage.output_tokens,
    }

    log.info(
        "llm_call_complete",
        component="prime",
        model=model,
        input_tokens=token_usage["input"],
        output_tokens=token_usage["output"],
    )

    return content, token_usage
