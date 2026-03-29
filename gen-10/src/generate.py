"""LLM integration — prompt building, API calls, response parsing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import structlog

logger = structlog.get_logger().bind(component="prime")


class ParseError(Exception):
    """Raised when LLM response cannot be parsed into file blocks."""
    pass


def parse_files(response: str) -> dict[str, str]:
    """
    Parse <file path="...">content</file:end> blocks from LLM response.
    Uses a line-by-line state machine to handle nested/embedded file tags.
    Stops at the FIRST </file:end> line encountered while inside a block.
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
- Test strings that embed XML-like content (e.g. <file> blocks) MUST use raw strings
  (r\"...\") or triple-quoted strings to avoid escaping issues.
"""


def build_fresh_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
) -> str:
    """Build the user message for a fresh generation attempt."""
    history_json = json.dumps(history, indent=2)
    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{history_json}\n\n"
        f"# Task\n\n"
        f"Produce a complete working codebase that implements the specification above.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_retry_prompt(
    spec_content: str,
    history: list[dict[str, Any]],
    generation: int,
    parent: int,
    failed_context: dict[str, Any],
) -> str:
    """Build the user message for a retry after failure."""
    history_json = json.dumps(history, indent=2)
    diagnostics = failed_context.get("diagnostics", {})
    failed_gen = failed_context.get("generation", generation - 1)
    stage = diagnostics.get("stage", "unknown")
    summary = diagnostics.get("summary", "unknown failure")

    failed_files_section = ""
    for file_path, content in failed_context.get("files", {}).items():
        ext = Path(file_path).suffix.lstrip(".")
        if not ext:
            ext = "text"
        failed_files_section += f"### {file_path}\n```{ext}\n{content}\n```\n\n"

    diagnostics_json = json.dumps(diagnostics, indent=2)

    return (
        f"# Specification\n\n{spec_content}\n\n"
        f"# Generation History\n\n{history_json}\n\n"
        f"# Previous Attempt Failed\n\n"
        f"Generation {failed_gen} failed at stage: {stage}\n"
        f"Summary: {summary}\n\n"
        f"## Failed Source Code\n\n{failed_files_section}"
        f"## Diagnostics\n\n{diagnostics_json}\n\n"
        f"# Task\n\n"
        f"The previous attempt failed. Study the failed code and diagnostics above.\n"
        f"Produce a complete, corrected codebase that fixes the identified issues.\n"
        f"Generation number: {generation}\n"
        f"Parent generation: {parent}\n"
    )


def build_parse_repair_prompt(
    raw_response: str,
    parse_error: str,
) -> str:
    """Build the user message for a parse repair attempt."""
    return (
        f"# Parse Error\n\n"
        f"The previous response could not be parsed. Error: {parse_error}\n\n"
        f"# Malformed Response\n\n{raw_response}\n\n"
        f"# Task\n\n"
        f"Re-emit the EXACT SAME files using the correct format. Every <file> block MUST have a\n"
        f"matching </file:end> on its own line. No nesting. No extra content between blocks.\n"
    )


@dataclass
class GenerationConfig:
    """Configuration for the generation loop."""
    anthropic_api_key: str
    model: str = "claude-sonnet-4-6"
    escalation_model: str = "claude-opus-4-6"
    max_gens: int = 5
    max_retries: int = 3
    max_parse_retries: int = 2
    spec_path: Path = field(default_factory=lambda: Path("./spec/CAMBRIAN-SPEC-005.md"))
    workspace_root: Path = field(default_factory=lambda: Path("/workspace"))


class LLMGenerator:
    """Handles LLM API calls for code generation."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    def _select_model(self, retry_count: int) -> str:
        """Select model based on retry count."""
        if retry_count == 0:
            return self.config.model
        return self.config.escalation_model

    async def _call_llm(
        self,
        model: str,
        user_message: str,
    ) -> tuple[str, dict[str, int]]:
        """Call the LLM API with streaming and return (text, token_usage)."""
        log = logger.bind(model=model)
        log.info("Calling LLM")

        async with self.client.messages.stream(
            model=model,
            max_tokens=16384,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            message = await stream.get_final_message()

        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        usage = {
            "input": message.usage.input_tokens,
            "output": message.usage.output_tokens,
        }
        log.info("LLM call complete", input_tokens=usage["input"], output_tokens=usage["output"])
        return text, usage

    async def generate(
        self,
        spec_content: str,
        history: list[dict[str, Any]],
        generation: int,
        parent: int,
        retry_count: int = 0,
        failed_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate a complete codebase from the spec."""
        model = self._select_model(retry_count)

        if failed_context is not None and retry_count > 0:
            user_message = build_retry_prompt(
                spec_content=spec_content,
                history=history,
                generation=generation,
                parent=parent,
                failed_context=failed_context,
            )
        else:
            user_message = build_fresh_prompt(
                spec_content=spec_content,
                history=history,
                generation=generation,
                parent=parent,
            )

        raw_response, token_usage = await self._call_llm(model=model, user_message=user_message)

        files = parse_files(raw_response)

        return {
            "files": files,
            "raw_response": raw_response,
            "token_usage": token_usage,
            "model": model,
        }

    async def repair(
        self,
        raw_response: str,
        parse_error: str,
        retry_count: int = 0,
    ) -> dict[str, Any]:
        """Attempt to repair a malformed LLM response."""
        model = self._select_model(retry_count)
        user_message = build_parse_repair_prompt(
            raw_response=raw_response,
            parse_error=parse_error,
        )
        repaired_response, token_usage = await self._call_llm(model=model, user_message=user_message)
        files = parse_files(repaired_response)
        return {
            "files": files,
            "raw_response": repaired_response,
            "token_usage": token_usage,
            "model": model,
        }