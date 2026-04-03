"""LLM integration: prompt building, API calls, response parsing."""
from __future__ import annotations

import os
import re
from typing import Any


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


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


SYSTEM_PROMPT = (
    "You are a code generator. You produce complete, working Python codebases from specifications.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY <file path=\"...\">content</file:end> blocks. One block per file.\n"
    "- Every file needed to build, test, and run the project must be in a <file> block.\n"
    "- Include a requirements.txt with all dependencies.\n"
    "- Include a test suite that exercises all functionality.\n"
    "- The code must work in Python 3.14 inside a Docker container with a venv at /venv.\n"
    "- Do NOT include manifest.json \u2014 it is generated separately.\n"
    "- Do NOT include the spec file \u2014 it is copied separately.\n"
    "- Python 3.14 STRICT: string literals MUST NOT contain unescaped newlines. Use\n"
    "  triple quotes (\"\"\" or ''') for multi-line strings. Use \\n for embedded newlines in\n"
    "  single-line strings. A bare newline inside \"...\" or '...' is a SyntaxError.\n"
    "- Test strings that embed XML-like content MUST use raw strings (r\"...\") or\n"
    "  triple-quoted strings to avoid escaping issues.\n"
)


def get_system_prompt() -> str:
    """Return the system prompt for the LLM."""
    return SYSTEM_PROMPT


def build_fresh_prompt(
    spec_content: str,
    generation_records_json: str,
    generation_number: int,
    parent_generation: int,
) -> str:
    """Build the user message for a fresh generation attempt."""
    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{generation_records_json}\n\n"
        f"# Task\n\n"
        f"Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation_number}\n"
        f"Parent generation: {parent_generation}\n"
    )


def build_retry_prompt(
    spec_content: str,
    generation_records_json: str,
    generation_number: int,
    parent_generation: int,
    prev_generation: int,
    diagnostics: dict[str, Any],
    failed_files: dict[str, str],
) -> str:
    """Build the user message for an informed retry."""
    files_section = ""
    for file_path, file_content in sorted(failed_files.items()):
        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lang = "python" if ext == "py" else ext
        files_section += f"### {file_path}\n```{lang}\n{file_content}\n```\n\n"

    import json
    diag_json = json.dumps(diagnostics, indent=2)

    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown error")

    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{generation_records_json}\n\n"
        f"# Previous Attempt Failed\n\n"
        f"Generation {prev_generation} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        f"## Failed Source Code\n\n{files_section}"
        f"## Diagnostics\n\n{diag_json}\n\n"
        f"# Task\n\n"
        f"The previous attempt failed. Study the failed code and diagnostics above.\n"
        f"Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation_number}\n"
        f"Parent generation: {parent_generation}\n"
    )


def build_parse_repair_prompt(
    parse_error_message: str,
    raw_llm_response: str,
) -> str:
    """Build the user message for a parse repair attempt."""
    return (
        f"# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error_message}\n\n"
        f"# Malformed Response\n\n{raw_llm_response}\n\n"
        f"# Task\n\n"
        f"Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        f"matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )


def select_model(retry_count: int) -> str:
    """Select the LLM model based on retry count."""
    if retry_count >= 1:
        return os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
    return os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")


def get_max_retries() -> int:
    """Get the maximum number of consecutive retries."""
    return int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))


def get_max_gens() -> int:
    """Get the maximum number of generation attempts."""
    return int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))


def get_max_parse_retries() -> int:
    """Get the maximum number of parse repair attempts."""
    return int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))


def compute_next_generation(records: list[dict[str, Any]]) -> int:
    """Compute the next generation number from history."""
    if not records:
        return 1
    max_gen = max(r.get("generation", 0) for r in records)
    return max_gen + 1
