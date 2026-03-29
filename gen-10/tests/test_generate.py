"""Tests for LLM prompt construction and response parsing."""

from __future__ import annotations

import pytest

from src.generate import (
    ParseError,
    build_fresh_prompt,
    build_parse_repair_prompt,
    build_retry_prompt,
    parse_files,
    SYSTEM_PROMPT,
    GenerationConfig,
)


def test_parse_single_file() -> None:
    """Parse a response with a single file block."""
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "hello.py" in files
    assert files["hello.py"] == 'print("hello")\n'


def test_parse_multiple_files() -> None:
    """Parse a response with multiple file blocks."""
    response = (
        '<file path="a.py">\ncontent_a\n</file:end>\n'
        '<file path="b.py">\ncontent_b\n</file:end>\n'
    )
    files = parse_files(response)
    assert "a.py" in files
    assert "b.py" in files
    assert files["a.py"] == "content_a\n"
    assert files["b.py"] == "content_b\n"


def test_parse_preserves_content_exactly() -> None:
    """File content is preserved exactly as-is."""
    content = "line1\nline2\n    indented\n"
    response = f'<file path="exact.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["exact.py"] == content


def test_parse_file_with_empty_content() -> None:
    """A file block with empty content produces an empty string."""
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at the FIRST </file:end> on its own line.
    Text mentioning </file:end> as part of a line is NOT the end marker.
    But a REAL </file:end> on its own line terminates the block,
    so content AFTER the first </file:end>-on-its-own-line is not included.
    """
    # This line contains "</file:end>" embedded in other text — not a terminator
    inner_line = "This mentions </file:end> in text but is not the real end\n"
    # This line IS just "</file:end>" followed by a newline — IS the terminator
    response = (
        '<file path="tricky.py">\n'
        + inner_line
        + "actual content\n"
        + "</file:end>\n"
    )
    files = parse_files(response)
    assert "tricky.py" in files
    # inner_line has text BEFORE </file:end>, so it's NOT the end marker
    # actual content comes BEFORE the real </file:end> line, so it IS included
    assert "actual content" in files["tricky.py"]
    assert inner_line in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are handled correctly."""
    response = '<file path="src/module/file.py">\ncontent\n</file:end>\n'
    files = parse_files(response)
    assert "src/module/file.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """An unclosed <file> block raises ParseError."""
    response = '<file path="unclosed.py">\ncontent\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message mentions the unclosed path."""
    with pytest.raises(ParseError, match="missing_end.py"):
        parse_files('<file path="missing_end.py">\ncontent\n')


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside <file> blocks is silently discarded."""
    response = (
        "Here is some commentary.\n"
        '<file path="real.py">\ncode\n</file:end>\n'
        "More commentary at the end.\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["real.py"]
    assert files["real.py"] == "code\n"


def test_parse_empty_response() -> None:
    """An empty response returns an empty dict."""
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """A response with no file blocks returns an empty dict."""
    files = parse_files("Just commentary, no files here.\n")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file path=...> tag inside file content does not open a new block."""
    inner_content = 'Example: <file path="inner.py">some content</file:end>\n'
    # The </file:end> here is embedded mid-line, not on its own line — not a terminator
    # But wait, the line is:
    #   'Example: <file path="inner.py">some content</file:end>\n'
    # rstrip("\n\r") = 'Example: <file path="inner.py">some content</file:end>'
    # That does NOT equal "</file:end>", so it's treated as content
    response = (
        '<file path="outer.py">\n'
        + inner_content
        + "</file:end>\n"
    )
    files = parse_files(response)
    assert "outer.py" in files
    assert "inner.py" not in files
    assert inner_content in files["outer.py"]


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    prompt = build_fresh_prompt(
        spec_content="MY SPEC CONTENT",
        history=[],
        generation=1,
        parent=0,
    )
    assert "MY SPEC CONTENT" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    history = [{"generation": 1, "outcome": "promoted"}]
    prompt = build_fresh_prompt(
        spec_content="spec",
        history=history,
        generation=2,
        parent=1,
    )
    assert "promoted" in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    prompt = build_fresh_prompt(
        spec_content="spec",
        history=[],
        generation=5,
        parent=4,
    )
    assert "5" in prompt
    assert "4" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics from failed context."""
    failed_context = {
        "generation": 3,
        "diagnostics": {
            "stage": "test",
            "summary": "5 tests failed",
            "exit_code": 1,
            "failures": [],
            "stdout_tail": "",
            "stderr_tail": "",
        },
        "files": {},
    }
    prompt = build_retry_prompt(
        spec_content="spec",
        history=[],
        generation=4,
        parent=3,
        failed_context=failed_context,
    )
    assert "5 tests failed" in prompt
    assert "test" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    failed_context = {
        "generation": 3,
        "diagnostics": {"stage": "test", "summary": "fail", "exit_code": 1,
                        "failures": [], "stdout_tail": "", "stderr_tail": ""},
        "files": {"src/main.py": "def broken(): pass\n"},
    }
    prompt = build_retry_prompt(
        spec_content="spec",
        history=[],
        generation=4,
        parent=3,
        failed_context=failed_context,
    )
    assert "def broken(): pass" in prompt
    assert "src/main.py" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the parse error message."""
    prompt = build_parse_repair_prompt(
        raw_response="bad response",
        parse_error="Unclosed block at path='foo.py'",
    )
    assert "Unclosed block at path='foo.py'" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    prompt = build_parse_repair_prompt(
        raw_response="THE RAW RESPONSE HERE",
        parse_error="some error",
    )
    assert "THE RAW RESPONSE HERE" in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses the base model."""
    from src.generate import LLMGenerator
    config = GenerationConfig(
        anthropic_api_key="test",
        model="claude-sonnet-4-6",
        escalation_model="claude-opus-4-6",
    )
    gen = LLMGenerator(config)
    assert gen._select_model(0) == "claude-sonnet-4-6"


def test_model_escalation_on_retry() -> None:
    """Retry uses the escalation model."""
    from src.generate import LLMGenerator
    config = GenerationConfig(
        anthropic_api_key="test",
        model="claude-sonnet-4-6",
        escalation_model="claude-opus-4-6",
    )
    gen = LLMGenerator(config)
    assert gen._select_model(1) == "claude-opus-4-6"
    assert gen._select_model(2) == "claude-opus-4-6"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key instructions."""
    assert "file path=" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
    assert "Python 3.14" in SYSTEM_PROMPT