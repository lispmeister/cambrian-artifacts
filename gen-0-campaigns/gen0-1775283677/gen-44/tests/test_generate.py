"""Tests for LLM prompt construction and response parsing."""
from __future__ import annotations

import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    result = parse_files(response)
    assert result == {"hello.py": 'print("hello")\n'}


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
    """File with empty content is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert "empty.py" in result
    assert result["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at FIRST </file:end> on its own line.
    Per spec Vector 1: </file:end> embedded in a longer line is NOT a close tag.
    But </file:end> alone on its own line ALWAYS closes the block.
    """
    from src.generate import parse_files
    # The line 'This mentions </file:end> in text...' has </file:end> NOT alone
    # so it's treated as content. But 'actual content' comes after the REAL
    # </file:end> (which IS alone on its own line), so it's discarded.
    # Actually per spec: the line "This mentions </file:end> in text..." does NOT
    # match because the ENTIRE line stripped is not "</file:end>".
    # So "actual content" IS in the file because the parser only stops at the
    # bare </file:end> line.
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # The state machine stops at the FIRST </file:end> on its own line
    # "This mentions </file:end> in text..." is NOT a close (not alone on line)
    # so actual content IS included
    assert "actual content" in files["tricky.py"]
    assert "This mentions </file:end> in text but is not the real end" in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are preserved."""
    from src.generate import parse_files
    response = '<file path="src/utils.py">\npass\n</file:end>\n'
    result = parse_files(response)
    assert "src/utils.py" in result


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    with pytest.raises(ParseError):
        parse_files('<file path="missing_end.py">\ncontent\n')


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed path."""
    from src.generate import ParseError, parse_files
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as e:
        assert "missing_end.py" in str(e)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is ignored."""
    from src.generate import parse_files
    response = (
        "Here is some commentary.\n"
        '<file path="a.py">\ncode\n</file:end>\n'
        "More commentary.\n"
    )
    result = parse_files(response)
    assert result == {"a.py": "code\n"}


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with only commentary returns empty dict."""
    from src.generate import parse_files
    result = parse_files("Just some text, no file blocks.")
    assert result == {}


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
    assert 'text with <file path="inner.py"> tag\n' == files["outer.py"]


def test_parse_spec_vector_1() -> None:
    """Spec Vector 1: </file:end> embedded in a longer line is NOT a close tag."""
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
    """Spec Vector 3: content after </file:end> before next <file> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    system_msg, user_msg = build_fresh_prompt(
        spec_content="MY SPEC CONTENT",
        generation_records=[],
        generation=1,
        parent=0,
    )
    assert "MY SPEC CONTENT" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    records = [{"generation": 1, "outcome": "promoted"}]
    system_msg, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        generation=2,
        parent=1,
    )
    assert "promoted" in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    system_msg, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        generation=42,
        parent=41,
    )
    assert "42" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    system_msg, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in user_msg
    assert "test" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    failed_files = {"src/prime.py": "def broken(): pass\n"}
    system_msg, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics={"stage": "test", "summary": "failed", "exit_code": 1,
                     "failures": [], "stdout_tail": "", "stderr_tail": ""},
        failed_files=failed_files,
    )
    assert "broken" in user_msg
    assert "src/prime.py" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    system_msg, user_msg = build_parse_repair_prompt(
        parse_error_message="Unclosed <file path='x.py'> block",
        raw_response="<file path='x.py'>\ncontent",
    )
    assert "Unclosed" in user_msg
    assert "x.py" in user_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file path='broken.py'>\nno end tag"
    system_msg, user_msg = build_parse_repair_prompt(
        parse_error_message="Unclosed block",
        raw_response=raw,
    )
    assert raw in user_msg


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL env var."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_MODEL": "claude-test-model"}):
        model = get_model(0)
    assert model == "claude-test-model"


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL env var."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation"}):
        model = get_model(1)
    assert model == "claude-opus-escalation"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
