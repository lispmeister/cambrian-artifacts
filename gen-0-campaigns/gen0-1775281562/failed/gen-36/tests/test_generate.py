"""Tests for LLM prompt construction and response parsing."""
from __future__ import annotations

import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nprint("hello")\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'print("hello")\n'}


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
    """Parse preserves whitespace and content exactly."""
    from src.generate import parse_files
    content = "line1\n  line2\n\nline4\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """Parse a file block with empty content."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    The close-tag match is EXACT: line.rstrip('\\n\\r') == '</file:end>'.
    A line containing '</file:end>' as a substring but with other characters
    does NOT close the block — those characters accumulate as content.
    The FIRST line that IS exactly '</file:end>' closes the block.

    Spec test vector 1:
    response = '<file path="a.py">\\nx = "</file:end>"\\n</file:end>\\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\\n'}
    """
    from src.generate import parse_files
    # Vector 1: </file:end> embedded in a longer line — NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_multiple_close_tags_in_content_with_prefix() -> None:
    """
    A line like 'This mentions </file:end> in text' is NOT a close tag
    because the whole stripped line != '</file:end>'. Content accumulates.
    The block closes at the FIRST line that IS exactly '</file:end>'.
    """
    from src.generate import parse_files
    # 'This mentions </file:end> in text' is NOT alone on its own line
    # so it does NOT close the block — it's accumulated as content
    response = (
        '<file path="tricky.py">\n'
        'This mentions </file:end> in text but is not the real end\n'
        'actual content\n'
        '</file:end>\n'
    )
    result = parse_files(response)
    assert "tricky.py" in result
    # Both lines are content because neither is EXACTLY '</file:end>'
    assert "This mentions </file:end> in text but is not the real end\n" in result["tricky.py"]
    assert "actual content\n" in result["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """Parse a file with a path containing subdirectories."""
    from src.generate import parse_files
    response = '<file path="src/sub/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/sub/module.py" in result
    assert result["src/sub/module.py"] == "content\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """ParseError is raised for unclosed file block."""
    from src.generate import parse_files, ParseError
    response = '<file path="missing_end.py">\ncontent\n'
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
    """Content outside file blocks is silently discarded."""
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
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    result = parse_files("Just some text, no file blocks here.\n")
    assert result == {}


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
    # The full content is accumulated (state machine is in 'inside block' state)
    assert 'text with <file path="inner.py"> tag\n' == result["outer.py"]


def test_parse_spec_vector_1() -> None:
    """Spec test vector 1: </file:end> embedded in longer line is NOT a close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Spec test vector 2: multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    assert parse_files(response) == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Spec test vector 3: content after </file:end> before next block is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    spec = "# My Spec\nSome content"
    _, user_msg = build_fresh_prompt(
        spec_content=spec,
        history=[],
        generation=1,
        parent=0,
    )
    assert spec in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history as JSON."""
    from src.generate import build_fresh_prompt
    history = [{"generation": 1, "outcome": "promoted"}]
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        history=history,
        generation=2,
        parent=1,
    )
    assert '"generation": 1' in user_msg
    assert '"outcome": "promoted"' in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        history=[],
        generation=5,
        parent=4,
    )
    assert "Generation number: 5" in user_msg
    assert "Parent generation: 4" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1}
    _, user_msg = build_retry_prompt(
        spec_content="spec",
        history=[],
        generation=2,
        parent=1,
        failed_gen=1,
        failed_files={},
        diagnostics=diagnostics,
    )
    assert "test" in user_msg
    assert "3 tests failed" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    failed_files = {"src/main.py": "def broken():\n    pass\n"}
    _, user_msg = build_retry_prompt(
        spec_content="spec",
        history=[],
        generation=2,
        parent=1,
        failed_gen=1,
        failed_files=failed_files,
        diagnostics={},
    )
    assert "src/main.py" in user_msg
    assert "def broken():" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the parse error message."""
    from src.generate import build_parse_repair_prompt
    error_msg = "Unclosed <file path='foo.py'> block"
    repair_msg = build_parse_repair_prompt(
        parse_error=error_msg,
        raw_response="some response",
    )
    assert error_msg in repair_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file path='unclosed'>\ncontent without end"
    repair_msg = build_parse_repair_prompt(
        parse_error="Unclosed block",
        raw_response=raw,
    )
    assert raw in repair_msg


def test_model_selection_first_attempt() -> None:
    """First attempt uses default model."""
    from src.generate import get_model
    import os
    original = os.environ.get("CAMBRIAN_MODEL")
    os.environ["CAMBRIAN_MODEL"] = "claude-sonnet-4-6"
    try:
        model = get_model(retry_count=0)
        assert model == "claude-sonnet-4-6"
    finally:
        if original is None:
            os.environ.pop("CAMBRIAN_MODEL", None)
        else:
            os.environ["CAMBRIAN_MODEL"] = original


def test_model_escalation_on_retry() -> None:
    """Retry attempts use escalation model."""
    from src.generate import get_model
    import os
    original = os.environ.get("CAMBRIAN_ESCALATION_MODEL")
    os.environ["CAMBRIAN_ESCALATION_MODEL"] = "claude-opus-4-6"
    try:
        model = get_model(retry_count=1)
        assert model == "claude-opus-4-6"
        model = get_model(retry_count=2)
        assert model == "claude-opus-4-6"
    finally:
        if original is None:
            os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
        else:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = original


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "file:end" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
