"""Tests for LLM integration: prompt building, response parsing."""
import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nhello\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": "hello\n"}


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
    """File content is preserved exactly."""
    from src.generate import parse_files
    content = "line1\nline2\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """Empty file block parses to empty string."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at FIRST </file:end> on its OWN line.
    A line like 'text </file:end> more text' does NOT close the block
    because the whole line is not exactly '</file:end>'.
    """
    from src.generate import parse_files
    # The line 'This mentions </file:end> in text' is NOT a close tag
    # because the stripped line != '</file:end>'
    response = (
        '<file path="tricky.py">\n'
        'This mentions </file:end> in text but is not the real end\n'
        'actual content\n'
        '</file:end>\n'
    )
    result = parse_files(response)
    assert "tricky.py" in result
    # Both lines are in the content because the first line has extra text
    assert "actual content" in result["tricky.py"]
    assert "This mentions </file:end> in text" in result["tricky.py"]


def test_parse_spec_vector_1() -> None:
    """Vector 1: </file:end> embedded in a longer line — NOT a close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Vector 2: Multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    assert parse_files(response) == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Vector 3: </file:end> alone closes the block; trailing content discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_parse_path_with_subdirectory() -> None:
    """Path with subdirectory is preserved."""
    from src.generate import parse_files
    response = '<file path="src/main.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/main.py" in result


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import parse_files, ParseError
    with pytest.raises(ParseError):
        parse_files('<file path="missing_end.py">\ncontent\n')


def test_parse_error_message_contains_path() -> None:
    """ParseError should mention the unclosed path."""
    from src.generate import parse_files, ParseError
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Commentary outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is the code:\n"
        '<file path="a.py">\ncontent\n</file:end>\n'
        "That's all!\n"
    )
    result = parse_files(response)
    assert result == {"a.py": "content\n"}


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    assert parse_files("") == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    assert parse_files("No files here, just text.") == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> open tag inside content doesn't start a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        'text with <file path="inner.py"> tag\n'
        '</file:end>\n'
    )
    result = parse_files(response)
    assert len(result) == 1
    assert "outer.py" in result
    assert 'text with <file path="inner.py"> tag\n' == result["outer.py"]


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("MY SPEC CONTENT", "[]", 1, 0)
    assert "MY SPEC CONTENT" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("spec", '[{"generation": 1}]', 2, 1)
    assert '"generation": 1' in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("spec", "[]", 5, 4)
    assert "5" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    _, user_msg = build_retry_prompt("spec", "[]", 2, 1, {}, diagnostics)
    assert "3 tests failed" in user_msg
    assert "test" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    failed_files = {"src/main.py": "def broken(): pass"}
    _, user_msg = build_retry_prompt("spec", "[]", 2, 1, failed_files,
                                     {"stage": "test", "summary": "failed"})
    assert "src/main.py" in user_msg
    assert "def broken(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    _, user_msg = build_parse_repair_prompt("Unclosed block at path='x.py'", "raw response")
    assert "Unclosed block at path='x.py'" in user_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    _, user_msg = build_parse_repair_prompt("some error", "THE RAW RESPONSE HERE")
    assert "THE RAW RESPONSE HERE" in user_msg


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_MODEL": "claude-test-model"}, clear=False):
        model = get_model(0)
    assert model == "claude-test-model"


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation"}, clear=False):
        model = get_model(1)
    assert model == "claude-opus-escalation"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains the file format rules."""
    from src.generate import SYSTEM_PROMPT
    assert "<file path=" in SYSTEM_PROMPT
    assert "</file:end>" in SYSTEM_PROMPT
