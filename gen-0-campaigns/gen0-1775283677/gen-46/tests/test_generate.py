"""Tests for LLM prompt construction and response parsing."""
import pytest


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
    """Content is preserved exactly including whitespace."""
    from src.generate import parse_files
    content = "line1\n    indented\nline3\n"
    response = f'<file path="test.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["test.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """Spec vector 1: </file:end> embedded in a longer line is NOT a close tag."""
    from src.generate import parse_files
    # When </file:end> appears embedded in a line (not alone), it's content
    # When </file:end> appears alone on its own line, it closes the block
    # The first standalone </file:end> closes the block
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """File path with subdirectory."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result
    assert result["src/module.py"] == "content\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
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
    except ParseError as exc:
        assert "missing_end.py" in str(exc)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Commentary outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is the code:\n"
        '<file path="a.py">\ncontent\n</file:end>\n'
        "That was the file.\n"
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
    result = parse_files("No files here, just commentary.\n")
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
    # The content includes the inner tag line fully
    assert 'text with <file path="inner.py"> tag\n' == files["outer.py"]


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
    """Spec vector 3: content after </file:end> before next <file> header is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    system_msg, user_msg = build_fresh_prompt(
        spec_content="THE SPEC",
        generation_records=[],
        offspring_gen=1,
        parent_gen=0,
    )
    assert "THE SPEC" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    records = [{"generation": 1, "outcome": "promoted"}]
    system_msg, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        offspring_gen=2,
        parent_gen=1,
    )
    assert "promoted" in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    from src.generate import build_fresh_prompt
    system_msg, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=5,
        parent_gen=4,
    )
    assert "5" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    system_msg, user_msg = build_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=2,
        parent_gen=1,
        diagnostics=diagnostics,
        failed_files={},
    )
    assert "3 tests failed" in user_msg
    assert "test" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "failure", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    failed_files = {"src/prime.py": "def main(): pass\n"}
    system_msg, user_msg = build_retry_prompt(
        spec_content="spec",
        generation_records=[],
        offspring_gen=2,
        parent_gen=1,
        diagnostics=diagnostics,
        failed_files=failed_files,
    )
    assert "src/prime.py" in user_msg
    assert "def main(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed block at 'foo.py'",
        raw_response="<file path='foo.py'>content",
    )
    assert "Unclosed block at 'foo.py'" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file path='foo.py'>content without end"
    prompt = build_parse_repair_prompt(
        parse_error_message="error",
        raw_response=raw,
    )
    assert raw in prompt


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
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
    assert "manifest.json" in SYSTEM_PROMPT
