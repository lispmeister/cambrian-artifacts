"""Tests for LLM prompt construction, response parsing, and parse repair."""
import pytest


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
    """File content is preserved exactly."""
    from src.generate import parse_files
    content = "line1\nline2\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File block with empty content is valid."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at FIRST </file:end> on its own line.
    A line containing </file:end> as a substring does NOT close the block.
    """
    from src.generate import parse_files
    # The line 'This mentions </file:end> in text but is not the real end'
    # is NOT equal to '</file:end>' when stripped — it should NOT close the block.
    # The next line 'actual content' should also be part of the file.
    # Then the real '</file:end>' on its own line closes the block.
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # The line 'This mentions </file:end> in text...' does NOT close the block
    # because the full stripped line != '</file:end>'.
    # So 'actual content' IS in the file.
    assert "actual content" in files["tricky.py"]
    assert "This mentions </file:end> in text but is not the real end" in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """File path with subdirectory is handled correctly."""
    from src.generate import parse_files
    response = '<file path="src/utils.py">\npass\n</file:end>\n'
    files = parse_files(response)
    assert "src/utils.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    response = '<file path="a.py">\ncontent without close tag\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message contains the unclosed path."""
    from src.generate import ParseError, parse_files
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "missing_end.py" in str(exc)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Content outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is the file:\n"
        '<file path="a.py">\ncode\n</file:end>\n'
        "And that's it.\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["a.py"]
    assert files["a.py"] == "code\n"


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    files = parse_files("This is just commentary without any file blocks.")
    assert files == {}


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
    """Spec vector 1: </file:end> embedded in a longer line is NOT a close tag."""
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
    """Spec vector 3: content after </file:end> before next <file> is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    messages = build_fresh_prompt(
        spec_content="SPEC CONTENT HERE",
        generation_records=[],
        offspring_gen=1,
        parent_gen=0,
    )
    assert len(messages) == 1
    assert "SPEC CONTENT HERE" in messages[0]["content"]


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    records = [{"generation": 1, "outcome": "promoted"}]
    messages = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        offspring_gen=2,
        parent_gen=1,
    )
    assert "promoted" in messages[0]["content"]


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    messages = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=5,
        parent_gen=4,
    )
    assert "5" in messages[0]["content"]


def test_retry_prompt_includes_diagnostics() -> None:
    """Informed retry prompt includes diagnostics."""
    from src.generate import build_informed_retry_prompt
    failed_context = {
        "diagnostics": {
            "stage": "test",
            "summary": "3 tests failed",
            "exit_code": 1,
            "failures": [],
            "stdout_tail": "",
            "stderr_tail": "",
        },
        "failed_files": {},
        "failed_generation": 2,
    }
    messages = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=3,
        parent_gen=2,
        failed_context=failed_context,
    )
    assert "3 tests failed" in messages[0]["content"]


def test_retry_prompt_includes_failed_code() -> None:
    """Informed retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    failed_context = {
        "diagnostics": {
            "stage": "test",
            "summary": "failure",
            "exit_code": 1,
            "failures": [],
            "stdout_tail": "",
            "stderr_tail": "",
        },
        "failed_files": {"src/main.py": "broken code here"},
        "failed_generation": 1,
    }
    messages = build_informed_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=2,
        parent_gen=1,
        failed_context=failed_context,
    )
    assert "broken code here" in messages[0]["content"]


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    messages = build_parse_repair_prompt(
        parse_error_message="Unclosed block at path 'foo.py'",
        raw_response="some raw response",
    )
    assert "Unclosed block at path 'foo.py'" in messages[0]["content"]


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    messages = build_parse_repair_prompt(
        parse_error_message="error",
        raw_response="RAW MALFORMED RESPONSE",
    )
    assert "RAW MALFORMED RESPONSE" in messages[0]["content"]


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
    """System prompt includes required rules."""
    from src.generate import SYSTEM_PROMPT
    assert "code generator" in SYSTEM_PROMPT
    assert "</file:end>" in SYSTEM_PROMPT
