"""Tests for LLM prompt construction and response parsing."""
from __future__ import annotations


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "a.py" in files
    assert files["a.py"] == 'print("hello")\n'


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    files = parse_files(response)
    assert files == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_preserves_content_exactly() -> None:
    """File content is preserved exactly as-is."""
    from src.generate import parse_files
    content = "line1\n  line2\n\tline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """A file block with no content produces an empty string."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """</file:end> embedded in a longer line is NOT a close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    files = parse_files(response)
    assert files == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """File paths can include subdirectories."""
    from src.generate import parse_files
    response = '<file path="src/main.py">\ncode\n</file:end>\n'
    files = parse_files(response)
    assert "src/main.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    import pytest
    with pytest.raises(ParseError):
        parse_files('<file path="bad.py">\ncontent\n')


def test_parse_error_message_contains_path() -> None:
    """ParseError message mentions the unclosed file path."""
    from src.generate import ParseError, parse_files
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is the code:\n\n"
        '<file path="a.py">\nfoo\n</file:end>\n'
        "\nSome more commentary.\n"
    )
    files = parse_files(response)
    assert files == {"a.py": "foo\n"}


def test_parse_empty_response() -> None:
    """Empty response produces no files."""
    from src.generate import parse_files
    assert parse_files("") == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks produces no files."""
    from src.generate import parse_files
    assert parse_files("Just some text\nNo file blocks here\n") == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> open tag inside content doesn't start a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        'text with <file path="inner.py"> tag\n'
        '</file:end>\n'
    )
    files = parse_files(response)
    assert len(files) == 1
    assert "outer.py" in files
    assert 'text with <file path="inner.py"> tag\n' in files["outer.py"]


def test_parse_spec_vector_1() -> None:
    """Spec vector 1: </file:end> embedded in longer line."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Spec vector 2: multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    assert parse_files(response) == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Spec vector 3: content after </file:end> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("my spec", "[]", 1, 0)
    assert "my spec" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", '[{"gen": 1}]', 2, 1)
    assert '{"gen": 1}' in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", "[]", 42, 41)
    assert "42" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diag = {"stage": "test", "summary": "3 tests failed"}
    prompt = build_retry_prompt("spec", "[]", 3, 2, 2, diag, {})
    assert "3 tests failed" in prompt
    assert "test" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diag = {"stage": "test", "summary": "fail"}
    failed = {"src/main.py": "print('hello')"}
    prompt = build_retry_prompt("spec", "[]", 3, 2, 2, diag, failed)
    assert "print('hello')" in prompt
    assert "src/main.py" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt("Unclosed block", "raw response")
    assert "Unclosed block" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw response."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt("error", "the raw LLM output")
    assert "the raw LLM output" in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses default model."""
    from src.generate import select_model
    import os
    old = os.environ.get("CAMBRIAN_MODEL")
    os.environ.pop("CAMBRIAN_MODEL", None)
    try:
        assert select_model(0) == "claude-sonnet-4-6"
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MODEL"] = old


def test_model_escalation_on_retry() -> None:
    """Retry uses escalation model."""
    from src.generate import select_model
    import os
    old = os.environ.get("CAMBRIAN_ESCALATION_MODEL")
    os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        assert select_model(1) == "claude-opus-4-6"
        assert select_model(2) == "claude-opus-4-6"
    finally:
        if old is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = old


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import get_system_prompt
    prompt = get_system_prompt()
    assert "code generator" in prompt
    assert "<file" in prompt
    assert "</file:end>" in prompt
    assert "requirements.txt" in prompt
