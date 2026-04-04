"""Tests for LLM prompt building and response parsing."""
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
    """Parser preserves file content exactly."""
    from src.generate import parse_files
    content = "line1\nline2\nline3\n"
    response = f'<file path="f.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["f.py"] == content


def test_parse_file_with_empty_content() -> None:
    """Parser handles files with empty content."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at the FIRST line that IS exactly </file:end>.
    A line that merely contains </file:end> as a substring (not the whole line)
    is NOT treated as a close tag per spec exact-match rule.
    """
    from src.generate import parse_files
    # Vector 1 from spec: </file:end> embedded in a longer line — NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """Parser handles paths with subdirectories."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result
    assert result["src/module.py"] == "content\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Parser raises ParseError for unclosed file blocks."""
    from src.generate import parse_files, ParseError
    response = '<file path="missing_end.py">\ncontent\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed path."""
    from src.generate import parse_files, ParseError
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "missing_end.py" in str(exc)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Content outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Some commentary here\n"
        '<file path="a.py">\ncontent\n</file:end>\n'
        "More commentary\n"
    )
    result = parse_files(response)
    assert result == {"a.py": "content\n"}


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    result = parse_files("This is just commentary with no file blocks.\n")
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
    assert files["outer.py"] == 'text with <file path="inner.py"> tag\n'


def test_parse_spec_vector_1() -> None:
    """Spec vector 1: </file:end> embedded in longer line is NOT a close tag."""
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
    """Spec vector 3: content after </file:end> before next <file> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="# My Spec",
        history_json="[]",
        generation=1,
        parent=0,
    )
    assert "# My Spec" in prompt["user"]


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="spec",
        history_json='[{"generation": 1}]',
        generation=2,
        parent=1,
    )
    assert '{"generation": 1}' in prompt["user"]


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="spec",
        history_json="[]",
        generation=42,
        parent=41,
    )
    assert "42" in prompt["user"]


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes failure diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1}
    prompt = build_retry_prompt(
        spec_content="spec",
        history_json="[]",
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "test" in prompt["user"]
    assert "3 tests failed" in prompt["user"]


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "build", "summary": "build failed"}
    failed_files = {"src/main.py": "def broken():\n    pass\n"}
    prompt = build_retry_prompt(
        spec_content="spec",
        history_json="[]",
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics=diagnostics,
        failed_files=failed_files,
    )
    assert "src/main.py" in prompt["user"]
    assert "def broken():" in prompt["user"]


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed <file path='x.py'> block",
        raw_response="<file path=\"x.py\">\ncontent",
    )
    assert "Unclosed" in prompt["user"]
    assert "x.py" in prompt["user"]


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = '<file path="x.py">\nsome content without closing tag'
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed block",
        raw_response=raw,
    )
    assert raw in prompt["user"]


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_MODEL": "claude-test-model"}):
        model = get_model(0)
    assert model == "claude-test-model"


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation"}):
        model = get_model(1)
    assert model == "claude-opus-escalation"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains required rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path" in SYSTEM_PROMPT
    assert "file:end" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
