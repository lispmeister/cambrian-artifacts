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
- Output ONLY <file path="...">content</file> blocks. One block per file.
- Every file needed to build, test, and run the project must be in a <file> block.
- Include a requirements.txt with all dependencies.
- Include a test suite that exercises all functionality.
- The code must work in Python 3.14 inside a Docker container with a venv at /venv.
- Do NOT include manifest.json — it is generated separately.
- Do NOT include the spec file — it is copied separately.
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


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def call_llm(system: str, user: str, model: str) -> tuple[str, int, int]:
    """Call the Anthropic API. Returns (response_text, input_tokens, output_tokens)."""
    client = anthropic.AsyncAnthropic()
    log.info("llm_call_start", model=model)

    message = await client.messages.create(
        model=model,
        max_tokens=32768,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = message.content[0].text
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    log.info("llm_call_complete", input_tokens=input_tokens, output_tokens=output_tokens)
    return text, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

FILE_PATTERN = re.compile(r'<file path="([^"]+)">(.*?)</file>', re.DOTALL)


def parse_files(response: str) -> dict[str, str]:
    """Extract files from <file path="...">content</file> blocks.

    Returns a dict mapping file paths to their contents.
    Strips leading/trailing newlines from each file's content.
    """
    matches = FILE_PATTERN.findall(response)
    if not matches:
        return {}
    return {path: content.strip("\n") for path, content in matches}
