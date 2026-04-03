"""Tests for LLM response parsing and prompt building."""
from __future__ import annotations


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="a.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": "content\n"}


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
    """File content is preserved exactly including whitespace."""
    from src.generate import parse_files
    content = "line1\nline2\n    indented\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content produces empty string."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine: </file:end> embedded mid-line is NOT a close tag.
    The close tag must be alone on its own line.

    Spec vector 1: response = '<file path="a.py">\\nx = "</file:end>"\\n</file:end>\\n'
    The line 'x = "</file:end>"' has </file:end> mid-line, NOT alone — not a close tag.
    The actual close tag is the next line '</file:end>' which IS alone.
    So the content includes 'x = "</file:end>"\\n'.
    """
    from src.generate import parse_files
    # Spec Vector 1: </file:end> embedded in a longer line — NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """Parse file with subdirectory path."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncode\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result
    assert result["src/module.py"] == "code\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """ParseError raised for unclosed file block."""
    from src.generate import ParseError, parse_files
    response = '<file path="missing_end.py">\ncontent\n'
    try:
        parse_files(response)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass


def test_parse_error_message_contains_path() -> None:
    """ParseError message contains the unclosed path."""
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
        "Here is the generated code:\n"
        '<file path="a.py">\ncode\n</file:end>\n'
        "And that's all!\n"
    )
    result = parse_files(response)
    assert result == {"a.py": "code\n"}


def test_parse_empty_response() -> None:
    """Empty response produces empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks produces empty dict."""
    from src.generate import parse_files
    result = parse_files("Just some commentary, no files.")
    assert result == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> tag inside file content doesn't open a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        '<file path="inner.py">\nnested content\n</file:end>\n'
        "continued outer\n"
        "</file:end>\n"
    )
    result = parse_files(response)
    # The state machine sees <file path="outer.py"> first.
    # Then it's in content-accumulation mode.
    # The <file path="inner.py"> line is treated as content (not a new block header).
    # The first </file:end> on its own line closes outer.py.
    assert "outer.py" in result
    assert "inner.py" not in result


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
    """Spec Vector 3: </file:end> alone closes the block; content after is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh generation prompt includes spec content."""
    from src.generate import build_fresh_prompt
    system, user = build_fresh_prompt(
        spec_content="THE SPEC CONTENT",
        generation_records=[],
        generation=1,
        parent=0,
    )
    assert "THE SPEC CONTENT" in user


def test_fresh_prompt_includes_history() -> None:
    """Fresh generation prompt includes generation history."""
    from src.generate import build_fresh_prompt
    records = [{"generation": 1, "outcome": "promoted"}]
    system, user = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        generation=2,
        parent=1,
    )
    assert '"generation": 1' in user
    assert '"outcome": "promoted"' in user


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh generation prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    system, user = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        generation=5,
        parent=4,
    )
    assert "5" in user
    assert "4" in user


def test_retry_prompt_includes_diagnostics() -> None:
    """Informed retry prompt includes diagnostics."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {
        "stage": "test",
        "summary": "3 tests failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    system, user = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in user
    assert "test" in user


def test_retry_prompt_includes_failed_code() -> None:
    """Informed retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {
        "stage": "build",
        "summary": "build failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    failed_files = {"src/prime.py": "def broken():\n    pass\n"}
    system, user = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics=diagnostics,
        failed_files=failed_files,
    )
    assert "src/prime.py" in user
    assert "def broken():" in user


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    _, user = build_parse_repair_prompt(
        parse_error_message="Unclosed block",
        raw_response="<file path='a.py'>\ncontent",
    )
    assert "Unclosed block" in user


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    _, user = build_parse_repair_prompt(
        parse_error_message="error",
        raw_response="MALFORMED_CONTENT_HERE",
    )
    assert "MALFORMED_CONTENT_HERE" in user


def test_model_selection_first_attempt() -> None:
    """First attempt uses base model."""
    from src.generate import get_model
    model = get_model("claude-sonnet-4-6", "claude-opus-4-6", retry_count=0)
    assert model == "claude-sonnet-4-6"


def test_model_escalation_on_retry() -> None:
    """Retry uses escalation model."""
    from src.generate import get_model
    model = get_model("claude-sonnet-4-6", "claude-opus-4-6", retry_count=1)
    assert model == "claude-opus-4-6"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "code generator" in SYSTEM_PROMPT.lower()
    assert "</file:end>" in SYSTEM_PROMPT
