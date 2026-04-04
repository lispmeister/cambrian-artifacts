"""Tests for LLM integration: prompt building and response parsing."""
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
    content = "line 1\n  line 2\n\nline 4\n"
    response = f'<file path="f.py">\n{content}</file:end>\n'
    result = parse_files(response)
    assert result["f.py"] == content


def test_parse_file_with_empty_content() -> None:
    """File with empty content is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at FIRST </file:end> on its own line.

    The spec says: A line containing </file:end> as a substring (not the whole line)
    does NOT close the block. A line that IS exactly </file:end> DOES close the block.

    According to Vector 1 in the spec:
      response = '<file path="a.py">\\nx = "</file:end>"\\n</file:end>\\n'
      assert parse_files(response) == {"a.py": 'x = "</file:end>"\\n'}

    This means: The line 'x = "</file:end>"' is NOT a close tag (substring, not whole line).
    The line '</file:end>' IS the close tag.

    For the tricky test: the inner line "This mentions </file:end> in text" has
    </file:end> as a substring, NOT the whole line. So it is NOT a close tag.
    The NEXT </file:end> on its own line IS the close tag.
    Therefore "actual content" IS in the file content (it comes before the real close tag).
    """
    from src.generate import parse_files
    # Vector 1 from spec: </file:end> embedded in a longer line — NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    assert parse_files(response) == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_1() -> None:
    """Spec Vector 1: </file:end> embedded in longer line is NOT a close tag."""
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
    """Spec Vector 3: Content after </file:end> is discarded (not ParseError)."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_parse_path_with_subdirectory() -> None:
    """File path with subdirectory is parsed correctly."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncode\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result
    assert result["src/module.py"] == "code\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import parse_files, ParseError
    response = '<file path="a.py">\ncontent without close tag\n'
    try:
        parse_files(response)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass


def test_parse_error_message_contains_path() -> None:
    """ParseError message contains the unclosed file path."""
    from src.generate import parse_files, ParseError
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
    result = parse_files("Just some commentary with no file blocks.")
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


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt user message includes spec content."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("MY SPEC CONTENT", "[]", 1, 0)
    assert "MY SPEC CONTENT" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt user message includes generation history."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("spec", '[{"generation": 1}]', 2, 1)
    assert '"generation": 1' in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt user message includes the generation number."""
    from src.generate import build_fresh_prompt
    _, user_msg = build_fresh_prompt("spec", "[]", 42, 41)
    assert "42" in user_msg
    assert "41" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics information."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    _, user_msg = build_retry_prompt("spec", "[]", 2, 1, diagnostics, {})
    assert "test" in user_msg
    assert "3 tests failed" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "failed", "exit_code": 1,
                   "failures": [], "stdout_tail": "", "stderr_tail": ""}
    failed_files = {"src/main.py": "def broken(): pass\n"}
    _, user_msg = build_retry_prompt("spec", "[]", 2, 1, diagnostics, failed_files)
    assert "src/main.py" in user_msg
    assert "def broken(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the parse error message."""
    from src.generate import build_parse_repair_prompt
    _, user_msg = build_parse_repair_prompt("Unclosed block at path='a.py'", "raw content")
    assert "Unclosed block at path='a.py'" in user_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    _, user_msg = build_parse_repair_prompt("some error", "MALFORMED RESPONSE HERE")
    assert "MALFORMED RESPONSE HERE" in user_msg


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
    assert "file path" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
