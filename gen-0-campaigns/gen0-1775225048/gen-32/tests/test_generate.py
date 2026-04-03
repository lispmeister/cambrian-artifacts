"""Tests for LLM prompt building and response parsing."""
from __future__ import annotations

import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nprint("hello")\n</file:end>\n'
    result = parse_files(response)
    assert "a.py" in result
    assert result["a.py"] == 'print("hello")\n'


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    result = parse_files(response)
    assert result == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_preserves_content_exactly() -> None:
    """Content between tags is preserved exactly."""
    from src.generate import parse_files
    content = "line1\nline2\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert "empty.py" in result
    assert result["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at FIRST </file:end> that is alone on its own line.
    A line containing </file:end> as part of a longer string does NOT close the block.

    Per spec Vector 1: '</file:end>' embedded in a longer line is NOT a close tag.
    The state machine stops at FIRST line where the entire stripped line == '</file:end>'.
    """
    from src.generate import parse_files
    # Vector 1 from spec: </file:end> embedded in a longer line — NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}

    # The state machine stops at FIRST </file:end> alone on its own line
    # "This mentions </file:end> in text" — the word "</file:end>" is embedded,
    # NOT alone on its own line, so it does NOT close the block.
    # The NEXT line "</file:end>" alone DOES close it.
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response2 = f'<file path="tricky.py">\n{inner}</file:end>\n'
    result2 = parse_files(response2)
    assert "tricky.py" in result2
    # The line "This mentions </file:end> in text..." is NOT a close tag (has more text)
    # So "actual content" IS included up to the real </file:end>
    assert "actual content" in result2["tricky.py"]
    assert "This mentions </file:end> in text but is not the real end" in result2["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """Paths with subdirectories are parsed correctly."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import parse_files, ParseError
    response = '<file path="missing_end.py">\ncontent without end\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message mentions the unclosed path."""
    from src.generate import parse_files, ParseError
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is some commentary\n"
        '<file path="a.py">\ncontent\n</file:end>\n'
        "More commentary\n"
    )
    result = parse_files(response)
    assert list(result.keys()) == ["a.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    result = parse_files("Just some text\nNo file blocks here\n")
    assert result == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """File tag inside content does not start a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        '<file path="inner.py">\n'
        'nested content\n'
        '</file:end>\n'
    )
    result = parse_files(response)
    # The inner <file path="inner.py"> is treated as content
    assert "outer.py" in result
    assert '<file path="inner.py">' in result["outer.py"]


def test_parse_spec_vector_1() -> None:
    """Spec Vector 1: </file:end> embedded in longer line is NOT a close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Spec Vector 2: Multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    assert parse_files(response) == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Spec Vector 3: Content after </file:end> before next <file> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("MY SPEC CONTENT", "[]", 1, 0)
    assert "MY SPEC CONTENT" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    history = '[{"generation": 1}]'
    prompt = build_fresh_prompt("spec", history, 2, 1)
    assert history in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", "[]", 42, 41)
    assert "42" in prompt
    assert "41" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diag = {"stage": "test", "summary": "3 tests failed", "exit_code": 1}
    prompt = build_retry_prompt("spec", "[]", 2, 1, 1, diag, {})
    assert "test" in prompt
    assert "3 tests failed" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    failed_files = {"src/prime.py": "def broken(): pass\n"}
    prompt = build_retry_prompt("spec", "[]", 2, 1, 1, {}, failed_files)
    assert "src/prime.py" in prompt
    assert "def broken(): pass" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_repair_prompt
    prompt = build_repair_prompt("Unclosed block at line 5", "raw response")
    assert "Unclosed block at line 5" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw response."""
    from src.generate import build_repair_prompt
    prompt = build_repair_prompt("error", "THE RAW RESPONSE")
    assert "THE RAW RESPONSE" in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses default model."""
    from src.generate import get_model, DEFAULT_MODEL
    assert get_model(0) == DEFAULT_MODEL


def test_model_escalation_on_retry() -> None:
    """Retry uses escalation model."""
    from src.generate import get_model, ESCALATION_MODEL
    assert get_model(1) == ESCALATION_MODEL
    assert get_model(2) == ESCALATION_MODEL


def test_system_prompt_contains_rules() -> None:
    """System prompt contains required rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "file:end" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
