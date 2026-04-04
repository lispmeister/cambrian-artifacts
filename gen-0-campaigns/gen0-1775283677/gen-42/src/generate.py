"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

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

    Uses a line-by-line state machine. The close tag </file:end> is matched
    ONLY when it is the entire line (stripped of newline chars). A line that
    contains </file:end> as a substring but has other content does NOT close
    the block.
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


def get_model(retry_count: int) -> str:
    """Get the LLM model to use based on retry count. Reads env vars at call time."""
    if retry_count == 0:
        return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    else:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")


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
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings\
 (r\"...\") or triple-quoted strings to avoid escaping issues.
- entry.start in manifest MUST use module form: python -m src.prime (NOT python src/prime.py).
- The src/ directory MUST contain __init__.py.
- structlog: first positional arg IS the event string. log.info("event_name", key=val).\
 NEVER: log.info("x", event="y") — that causes TypeError.
- Imports inside generation loop code MUST use absolute imports: from src.loop import ...\
 NOT relative imports when running as python -m src.prime.
"""


def build_fresh_prompt(
    spec_content: str,
    generation_history: str,
    generation: int,
    parent: int,
) -> tuple[str, str]:
    """Build the fresh generation prompt."""
    user_msg = f"""# Specification

{spec_content}

# Generation History

{generation_history}

# Task

Produce a complete working codebase that implements the specification above.
Generation number: {generation}
Parent generation: {parent}
"""
    return SYSTEM_PROMPT, user_msg


def build_retry_prompt(
    spec_content: str,
    generation_history: str,
    generation: int,
    parent: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> tuple[str, str]:
    """Build the informed retry prompt with failure context."""
    import json

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")
    diagnostics_json = json.dumps(diagnostics, indent=2)

    files_section = ""
    for file_path, content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        files_section += f"\n### {file_path}\n```{ext}\n{content}\n```\n"

    user_msg = f"""# Specification

{spec_content}

# Generation History

{generation_history}

# Previous Attempt Failed

Generation {parent} failed at stage: {stage}
Summary: {summary}

## Failed Source Code
{files_section}

## Diagnostics

```json
{diagnostics_json}
```

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
    """Build parse repair prompt."""
    user_msg = f"""# Parse Error

The previous response could not be parsed. Error: {parse_error_message}

# Malformed Response

{raw_response}

# Task

Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a
matching </file:end> on its own line. No nesting. No extra content between blocks.
"""
    return SYSTEM_PROMPT, user_msg


async def call_llm(
    model: str,
    system_message: str,
    user_message: str,
) -> tuple[str, dict[str, int]]:
    """Call the Anthropic LLM API using streaming."""
    import anthropic

    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    log.info("llm_call_starting", component="prime", model=model)

    async with client.messages.stream(
        model=model,
        max_tokens=16000,
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

    log.info("llm_call_complete", component="prime", model=model,
             input_tokens=token_usage["input"], output_tokens=token_usage["output"])

    return content, token_usage
