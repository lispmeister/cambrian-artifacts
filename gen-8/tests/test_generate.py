"""Tests for LLM prompt construction, response parsing."""

import pytest
from typing import Any

from src.generate import (
    parse_files,
    ParseError,
    build_fresh_prompt,
    build_retry_prompt,
    build_parse_repair_prompt,
    select_model,
    SYSTEM_PROMPT,
)


def test_parse_single_file() -> None:
    """Parse a single file block."""
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "hello.py" in files
    assert 'print("hello")\n' in files["hello.py"]


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks."""
    response = (
        '<file path="a.py">\ncontent_a\n</file:end>\n'
        '<file path="b.py">\ncontent_b\n</file:end>\n'
    )
    files = parse_files(response)
    assert "a.py" in files
    assert "b.py" in files
    assert "content_a" in files["a.py"]
    assert "content_b" in files["b.py"]


def test_parse_preserves_content_exactly() -> None:
    """Content between tags is preserved exactly."""
    content = "line1\nline2\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """State machine stops at FIRST </file:end> on its own line."""
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # The state machine stops at the FIRST </file:end> on its own line
    assert "actual content" not in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """Paths with subdirectories are parsed correctly."""
    response = '<file path="src/module.py">\ncode\n</file:end>\n'
    files = parse_files(response)
    assert "src/module.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    response = '<file path="unclosed.py">\ncontent\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError should mention the unclosed path."""
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    response = (
        "Some commentary here\n"
        '<file path="code.py">\nprint(1)\n</file:end>\n'
        "More commentary\n"
    )
    files = parse_files(response)
    assert len(files) == 1
    assert "code.py" in files


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    files = parse_files("Just some text with no file blocks")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file path=...> tag inside file content does not start a new block."""
    inner_content = 'This file contains a <file path="nested.py"> reference\n'
    response = f'<file path="outer.py">\n{inner_content}</file:end>\n'
    files = parse_files(response)
    assert len(files) == 1
    assert "outer.py" in files
    assert "nested.py" not in files


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    prompt = build_fresh_prompt(
        spec_content="MY SPEC",
        generation_records=[],
        generation_number=1,
        parent_generation=0,
    )
    assert "MY SPEC" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    records = [{"generation": 1, "outcome": "promoted"}]
    prompt = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        generation_number=2,
        parent_generation=1,
    )
    assert '"generation": 1' in prompt
    assert '"outcome": "promoted"' in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    prompt = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        generation_number=42,
        parent_generation=41,
    )
    assert "42" in prompt
    assert "41" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    diag = {"stage": "test", "summary": "5 tests failed", "exit_code": 1, "failures": []}
    prompt = build_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation_number=2,
        parent_generation=1,
        failed_gen_number=1,
        diagnostics=diag,
        failed_files={},
    )
    assert "test" in prompt
    assert "5 tests failed" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    failed_files = {"src/main.py": "def broken(): pass\n"}
    prompt = build_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation_number=2,
        parent_generation=1,
        failed_gen_number=1,
        diagnostics={"stage": "build", "summary": "failed", "exit_code": 1, "failures": []},
        failed_files=failed_files,
    )
    assert "src/main.py" in prompt
    assert "def broken(): pass" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes error message."""
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed block at path='foo.py'",
        raw_llm_response='<file path="foo.py">\ncontent\n',
    )
    assert "Unclosed block at path='foo.py'" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw response."""
    raw = '<file path="foo.py">\ncontent without end\n'
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed",
        raw_llm_response=raw,
    )
    assert "foo.py" in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL (default claude-sonnet-4-6)."""
    import os
    os.environ.pop("CAMBRIAN_MODEL", None)
    model = select_model(retry_count=0)
    assert model == "claude-sonnet-4-6"


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL (default claude-opus-4-6)."""
    import os
    os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    model = select_model(retry_count=1)
    assert model == "claude-opus-4-6"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    assert "code generator" in SYSTEM_PROMPT.lower()
    assert "</file:end>" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT