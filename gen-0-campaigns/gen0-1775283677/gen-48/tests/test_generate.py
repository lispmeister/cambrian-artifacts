"""Tests for LLM integration: prompt building and response parsing."""

from __future__ import annotations

import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "hello.py" in files
    assert files["hello.py"] == 'print("hello")\n'


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
    """File content is preserved exactly including whitespace."""
    from src.generate import parse_files
    content = "line1\n    indented\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    The state machine stops at the FIRST line that is EXACTLY </file:end>.
    A line containing </file:end> as part of a longer string does NOT close the block.
    """
    from src.generate import parse_files
    # The inner content has </file:end> embedded in a longer line (not exact match)
    # followed by actual content, followed by the real close tag on its own line.
    # Per spec Vector 1: '</file:end>' embedded in a longer line is NOT a close tag.
    # The real close tag is on its own line.
    response = (
        '<file path="a.py">\n'
        'x = "</file:end>"\n'
        '</file:end>\n'
    )
    files = parse_files(response)
    assert files == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are preserved."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\npass\n</file:end>\n'
    files = parse_files(response)
    assert "src/module.py" in files
    assert files["src/module.py"] == "pass\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    response = '<file path="unclosed.py">\ncontent\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed path."""
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
        "Here is the code:\n"
        '<file path="a.py">\npass\n</file:end>\n'
        "And that's the file.\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["a.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    files = parse_files("This is just text, no file blocks.")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> open tag inside content is treated as content, not a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        'text with <file path="inner.py"> tag\n'
        '</file:end>\n'
    )
    files = parse_files(response)
    assert len(files) == 1
    assert "outer.py" in files
    assert files["outer.py"] == 'text with <file path="inner.py"> tag\n'


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
    """Spec Vector 3: </file:end> alone on its own line closes the block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt(
        spec_content="MY SPEC CONTENT",
        generation_records=[],
        offspring_gen=1,
        parent_gen=0,
    )
    assert "MY SPEC CONTENT" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    history = [{"generation": 1, "outcome": "promoted"}]
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=history,
        offspring_gen=2,
        parent_gen=1,
    )
    assert '"generation": 1' in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=5,
        parent_gen=4,
    )
    assert "5" in user_msg
    assert "4" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes failure diagnostics."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1}
    _, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=3,
        parent_gen=2,
        failed_files={},
        diagnostics=diagnostics,
    )
    assert "3 tests failed" in user_msg
    assert "test" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    failed_files = {"src/prime.py": "def broken(): pass\n"}
    _, user_msg = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=3,
        parent_gen=2,
        failed_files=failed_files,
        diagnostics={"stage": "test", "summary": "failed"},
    )
    assert "src/prime.py" in user_msg
    assert "def broken(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    _, user_msg = build_parse_repair_prompt(
        parse_error_message="Unclosed block at path='foo.py'",
        raw_response="<file path='foo.py'>content",
    )
    assert "Unclosed block at path='foo.py'" in user_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file path='foo.py'>content without close"
    _, user_msg = build_parse_repair_prompt(
        parse_error_message="error",
        raw_response=raw,
    )
    assert raw in user_msg


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
    """System prompt contains key rules."""
    from src.generate import _get_system_prompt
    prompt = _get_system_prompt()
    assert "file path" in prompt
    assert "requirements.txt" in prompt
    assert "manifest.json" in prompt
