"""Tests for LLM prompt construction, response parsing, and model selection."""
from __future__ import annotations

import pytest

from src.generate import (
    CAMBRIAN_MAX_PARSE_RETRIES,
    CAMBRIAN_MAX_RETRIES,
    ParseError,
    build_fresh_prompt,
    build_informed_retry_prompt,
    build_parse_repair_prompt,
    get_generation_number,
    parse_files,
    select_model,
    SYSTEM_PROMPT,
    CAMBRIAN_MODEL,
    CAMBRIAN_ESCALATION_MODEL,
)


# ---------------------------------------------------------------------------
# parse_files tests
# ---------------------------------------------------------------------------

def test_parse_single_file() -> None:
    """Single file block is parsed correctly."""
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "hello.py" in files
    assert 'print("hello")' in files["hello.py"]


def test_parse_multiple_files() -> None:
    """Multiple file blocks are parsed correctly."""
    response = (
        '<file path="a.py">\ncontents_a\n</file:end>\n'
        '<file path="b.py">\ncontents_b\n</file:end>\n'
    )
    files = parse_files(response)
    assert "a.py" in files
    assert "b.py" in files
    assert "contents_a" in files["a.py"]
    assert "contents_b" in files["b.py"]


def test_parse_preserves_content_exactly() -> None:
    """File content is preserved exactly including whitespace."""
    content = "line1\nline2\n    indented\n"
    response = f'<file path="exact.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["exact.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """State machine stops at the FIRST </file:end> on its own line."""
    # The spec says: "The state machine stops at the FIRST </file:end> on its own line"
    # So if </file:end> appears on its own line inside content, it terminates the block.
    # This means the content after the first </file:end> is NOT part of the file.
    response = '<file path="tricky.py">\nsome content\n</file:end>\nextra content\n'
    files = parse_files(response)
    assert "tricky.py" in files
    assert "some content" in files["tricky.py"]
    # The state machine stops at the first </file:end> on its own line
    assert "extra content" not in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are handled correctly."""
    response = '<file path="src/sub/module.py">\ncontent\n</file:end>\n'
    files = parse_files(response)
    assert "src/sub/module.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    response = '<file path="unclosed.py">\ncontent without end\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message contains the unclosed path."""
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        pytest.fail("Expected ParseError")
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    response = (
        "Some commentary here\n"
        '<file path="code.py">\nprint("hi")\n</file:end>\n'
        "More commentary\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["code.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with only commentary returns empty dict."""
    files = parse_files("Just some text\nNo file blocks here\n")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> tag in content doesn't nest — state machine handles it."""
    # When already inside a file block, a <file path=...> line is treated as content
    inner_content = 'Here is a <file path="nested.py"> tag in content\nmore content\n'
    response = f'<file path="outer.py">\n{inner_content}</file:end>\n'
    files = parse_files(response)
    assert "outer.py" in files
    # The inner <file path=...> tag is part of the content since we're already in a block
    assert "nested.py" not in files
    assert "Here is a" in files["outer.py"]


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt user message includes spec content."""
    spec = "# Test Spec\nSome content"
    _, user_msg = build_fresh_prompt(
        spec_content=spec,
        generation_records=[],
        offspring_gen=1,
        parent_gen=0,
    )
    assert spec in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt user message includes generation history."""
    records = [{"generation": 1, "outcome": "promoted"}]
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        offspring_gen=2,
        parent_gen=1,
    )
    assert '"generation": 1' in user_msg
    assert '"outcome": "promoted"' in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt user message includes the generation number."""
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=5,
        parent_gen=4,
    )
    assert "Generation number: 5" in user_msg
    assert "Parent generation: 4" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Informed retry prompt includes diagnostics."""
    diagnostics = {
        "stage": "test",
        "summary": "3 tests failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    _, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=2,
        parent_gen=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in user_msg
    assert "test" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Informed retry prompt includes failed source code."""
    failed_files = {"src/main.py": "def broken(): pass\n"}
    _, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=2,
        parent_gen=1,
        diagnostics={"stage": "test", "summary": "failed", "exit_code": 1,
                     "failures": [], "stdout_tail": "", "stderr_tail": ""},
        failed_files=failed_files,
    )
    assert "src/main.py" in user_msg
    assert "def broken(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    msg = build_parse_repair_prompt("Unclosed block error", "some raw response")
    assert "Unclosed block error" in msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    raw = "some malformed <file> content"
    msg = build_parse_repair_prompt("error", raw)
    assert raw in msg


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL."""
    model = select_model(0)
    assert model == CAMBRIAN_MODEL


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL."""
    model = select_model(1)
    assert model == CAMBRIAN_ESCALATION_MODEL
    model2 = select_model(5)
    assert model2 == CAMBRIAN_ESCALATION_MODEL


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    assert "code generator" in SYSTEM_PROMPT
    assert "<file path=" in SYSTEM_PROMPT
    assert "</file:end>" in SYSTEM_PROMPT
