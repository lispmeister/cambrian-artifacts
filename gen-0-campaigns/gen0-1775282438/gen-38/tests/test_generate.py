"""Tests for LLM prompt construction and response parsing."""
import pytest


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nhello\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": "hello\n"}


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks in sequence."""
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
    response = f'<file path="f.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["f.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert "empty.py" in result
    assert result["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    The state machine stops at the FIRST </file:end> line.
    A line containing </file:end> as substring (not the whole line) does NOT close the block.
    """
    from src.generate import parse_files
    # Vector 1 from spec: </file:end> embedded in a longer line is NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are handled."""
    from src.generate import parse_files
    response = '<file path="src/foo.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/foo.py" in result
    assert result["src/foo.py"] == "content\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """An unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    with pytest.raises(ParseError):
        parse_files('<file path="a.py">\ncontent without closing tag\n')


def test_parse_error_message_contains_path() -> None:
    """ParseError message contains the unclosed file path."""
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
        "Here is my implementation:\n"
        '<file path="a.py">\ncode\n</file:end>\n'
        "That's all!\n"
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
    result = parse_files("Just some commentary, no files here.")
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
    """Spec vector 1: </file:end> embedded in longer line is NOT close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Spec vector 2: Multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    assert parse_files(response) == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Spec vector 3: Content after </file:end> before next <file> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt user message includes the spec content."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="my spec here",
        history=[],
        generation=1,
        parent=0,
    )
    assert "my spec here" in prompt["user"]


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    history = [{"generation": 1, "outcome": "promoted"}]
    prompt = build_fresh_prompt(
        spec_content="spec",
        history=history,
        generation=2,
        parent=1,
    )
    assert "promoted" in prompt["user"]


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="spec",
        history=[],
        generation=5,
        parent=4,
    )
    assert "5" in prompt["user"]


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    prompt = build_informed_retry_prompt(
        spec_content="spec",
        history=[],
        generation=2,
        parent=1,
        failed_gen=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in prompt["user"]
    assert "test" in prompt["user"]


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {"stage": "test", "summary": "failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    failed_files = {"src/main.py": "def broken(): pass\n"}
    prompt = build_informed_retry_prompt(
        spec_content="spec",
        history=[],
        generation=2,
        parent=1,
        failed_gen=1,
        diagnostics=diagnostics,
        failed_files=failed_files,
    )
    assert "def broken(): pass" in prompt["user"]
    assert "src/main.py" in prompt["user"]


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed block at 'foo.py'",
        raw_response="<file path='foo.py'>\nno end tag",
    )
    assert "Unclosed block at 'foo.py'" in prompt["user"]


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file path='foo.py'>\nno end tag"
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed",
        raw_response=raw,
    )
    assert raw in prompt["user"]


def test_model_selection_first_attempt() -> None:
    """First attempt uses default model."""
    import os
    from src.generate import select_model
    default = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    assert select_model(0) == default


def test_model_escalation_on_retry() -> None:
    """Retry uses escalation model."""
    import os
    from src.generate import select_model
    escalation = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
    assert select_model(1) == escalation
    assert select_model(2) == escalation


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
    assert "Python 3.14" in SYSTEM_PROMPT
