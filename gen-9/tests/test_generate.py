"""Tests for LLM prompt construction and response parsing."""
from __future__ import annotations

import pytest

from src.generate import (
    ParseError,
    build_fresh_prompt,
    build_parse_repair_prompt,
    build_retry_prompt,
    parse_files,
    select_model,
)


def test_parse_single_file() -> None:
    """Parse a single file block."""
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "hello.py" in files
    assert 'print("hello")' in files["hello.py"]


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks."""
    response = (
        '<file path="a.py">\ncontent_a\n</file:end>\n'
        '<file path="b.py">\ncontent_b\n</file:end>\n'
    )
    files = parse_files(response)
    assert set(files.keys()) == {"a.py", "b.py"}
    assert "content_a" in files["a.py"]
    assert "content_b" in files["b.py"]


def test_parse_preserves_content_exactly() -> None:
    """File content is preserved exactly including whitespace."""
    content = "line1\nline2\n    indented\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """Empty file block produces empty string."""
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    A line that CONTAINS </file:end> but has other text is kept as content.
    Only a line that IS exactly '</file:end>' terminates the block.
    """
    # This line contains </file:end> but is NOT on its own line
    inner = "This mentions </file:end> in text but continues on same line with more text\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # The embedded </file:end> mid-line does NOT terminate parsing
    # So actual content IS in the file
    assert "actual content" in files["tricky.py"]
    # The line with embedded </file:end> is also present
    assert "This mentions" in files["tricky.py"]


def test_parse_close_tag_on_own_line_terminates() -> None:
    """A line that is exactly </file:end> terminates the block."""
    response = '<file path="a.py">\nfirst\n</file:end>\nextra content not captured\n'
    files = parse_files(response)
    assert "a.py" in files
    assert "extra content" not in files["a.py"]
    assert "first" in files["a.py"]


def test_parse_path_with_subdirectory() -> None:
    """Path with subdirectory is preserved."""
    response = '<file path="src/module.py">\ncode\n</file:end>\n'
    files = parse_files(response)
    assert "src/module.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    response = '<file path="unclosed.py">\ncontent without end\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed file path."""
    response = '<file path="missing_end.py">\ncontent\n'
    with pytest.raises(ParseError) as exc_info:
        parse_files(response)
    assert "missing_end.py" in str(exc_info.value)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    response = (
        "Here is my explanation:\n"
        '<file path="code.py">\nprint("hi")\n</file:end>\n'
        "That was the code.\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["code.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    files = parse_files("This is just commentary with no file blocks.")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """
    A <file path="..."> tag inside file content does not start a new block.
    The state machine only looks for opening tags when not inside a block.
    """
    inner_tag = '<file path="nested.py">\nnested content\n</file:end>\n'
    response = f'<file path="outer.py">\n{inner_tag}real end\n</file:end>\n'
    files = parse_files(response)
    # The inner </file:end> terminates the outer block (it's on its own line)
    # So outer.py gets the inner tag line and 'nested content' line
    assert "outer.py" in files


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes the spec content."""
    prompt = build_fresh_prompt(
        spec_content="# My Spec\nContent here",
        generation=1,
        parent=0,
        history=[],
    )
    assert "# My Spec" in prompt
    assert "Content here" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    history = [{"generation": 1, "outcome": "promoted"}]
    prompt = build_fresh_prompt(
        spec_content="spec",
        generation=2,
        parent=1,
        history=history,
    )
    assert '"generation": 1' in prompt
    assert '"outcome": "promoted"' in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    prompt = build_fresh_prompt(
        spec_content="spec",
        generation=5,
        parent=4,
        history=[],
    )
    assert "Generation number: 5" in prompt
    assert "Parent generation: 4" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics information."""
    diagnostics = {
        "stage": "test",
        "summary": "3 tests failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    prompt = build_retry_prompt(
        spec_content="spec",
        generation=3,
        parent=2,
        history=[],
        failed_generation=2,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in prompt
    assert "stage" in prompt
    assert "test" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    failed_files = {
        "src/main.py": "def broken(): pass",
        "tests/test_main.py": "def test_broken(): assert False",
    }
    prompt = build_retry_prompt(
        spec_content="spec",
        generation=3,
        parent=2,
        history=[],
        failed_generation=2,
        diagnostics={"stage": "test", "summary": "failed"},
        failed_files=failed_files,
    )
    assert "src/main.py" in prompt
    assert "def broken(): pass" in prompt
    assert "tests/test_main.py" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed <file path='foo.py'> block",
        raw_response="some malformed content",
    )
    assert "Unclosed" in prompt
    assert "foo.py" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    prompt = build_parse_repair_prompt(
        parse_error_message="some error",
        raw_response="malformed response content here",
    )
    assert "malformed response content here" in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL (default claude-sonnet-4-6)."""
    import os
    orig = os.environ.pop("CAMBRIAN_MODEL", None)
    try:
        model = select_model(0)
        assert model == "claude-sonnet-4-6"
    finally:
        if orig is not None:
            os.environ["CAMBRIAN_MODEL"] = orig


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL (default claude-opus-4-6)."""
    import os
    orig = os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        model = select_model(1)
        assert model == "claude-opus-4-6"
        model = select_model(2)
        assert model == "claude-opus-4-6"
    finally:
        if orig is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = orig


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "code generator" in SYSTEM_PROMPT.lower()
    assert "</file:end>" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT