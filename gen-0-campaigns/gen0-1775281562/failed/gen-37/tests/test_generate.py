"""Tests for LLM prompt construction and response parsing."""


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
    """A file block with empty content produces an empty string."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert result == {"empty.py": ""}


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine: </file:end> only closes when it is the ENTIRE line.
    A line like 'This mentions </file:end> in text' is NOT a close tag
    because stripped it is NOT equal to </file:end>.
    So 'actual content' IS included in the output.

    Spec vector 1: response = '<file path="a.py">\\nx = "</file:end>"\\n</file:end>\\n'
    The line 'x = "</file:end>"' stripped is 'x = "</file:end>"' != '</file:end>'
    so it's accumulated as content.
    """
    from src.generate import parse_files

    # Test with embedded </file:end> inside a longer line
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}

    # The 'This mentions...' line - stripped is NOT equal to </file:end>
    # so it and 'actual content' are both accumulated
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response2 = f'<file path="tricky.py">\n{inner}</file:end>\n'
    result2 = parse_files(response2)
    assert "tricky.py" in result2
    # actual content IS included because the inner line was not a close tag
    assert "actual content" in result2["tricky.py"]
    assert "This mentions </file:end> in text" in result2["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """File paths with subdirectories are preserved."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """An unclosed file block raises ParseError."""
    from src.generate import ParseError, parse_files
    response = '<file path="missing_end.py">\ncontent\n'
    try:
        parse_files(response)
        assert False, "Expected ParseError"
    except ParseError:
        pass


def test_parse_error_message_contains_path() -> None:
    """ParseError should mention the unclosed path."""
    from src.generate import ParseError, parse_files
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Expected ParseError"
    except ParseError as exc:
        assert "missing_end.py" in str(exc)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Content outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is my commentary.\n"
        '<file path="a.py">\ncontent\n</file:end>\n'
        "More commentary.\n"
    )
    result = parse_files(response)
    assert list(result.keys()) == ["a.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    from src.generate import parse_files
    result = parse_files("Just some text with no file blocks.")
    assert result == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> open tag inside content is accumulated as content (state machine is in 'inside' state)."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        'text with <file path="inner.py"> tag\n'
        '</file:end>\n'
    )
    result = parse_files(response)
    assert len(result) == 1
    assert "outer.py" in result
    # The inner <file> tag line is accumulated as content
    assert 'text with <file path="inner.py"> tag\n' == result["outer.py"]


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
    prompt = build_fresh_prompt("MY SPEC CONTENT", "[]", 1, 0)
    assert "MY SPEC CONTENT" in prompt["user"]


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    history = '[{"generation": 1}]'
    prompt = build_fresh_prompt("spec", history, 2, 1)
    assert history in prompt["user"]


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", "[]", 42, 41)
    assert "42" in prompt["user"]


def test_retry_prompt_includes_diagnostics() -> None:
    """Informed retry prompt includes diagnostics."""
    from src.generate import build_informed_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1, "failures": []}
    prompt = build_informed_retry_prompt("spec", "[]", 2, 1, 1, diagnostics, {})
    assert "test" in prompt["user"]
    assert "3 tests failed" in prompt["user"]


def test_retry_prompt_includes_failed_code() -> None:
    """Informed retry prompt includes failed source code."""
    from src.generate import build_informed_retry_prompt
    failed_files = {"src/main.py": "def broken(): pass\n"}
    diagnostics = {"stage": "test", "summary": "failed", "exit_code": 1, "failures": []}
    prompt = build_informed_retry_prompt("spec", "[]", 2, 1, 1, diagnostics, failed_files)
    assert "src/main.py" in prompt["user"]
    assert "def broken(): pass" in prompt["user"]


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt("Unclosed block at path='foo.py'", "raw response")
    assert "Unclosed block at path='foo.py'" in prompt["user"]


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw malformed response."""
    from src.generate import build_parse_repair_prompt
    raw = "<file unclosed content here"
    prompt = build_parse_repair_prompt("some error", raw)
    assert raw in prompt["user"]


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL (default claude-sonnet-4-6)."""
    import os
    from src.generate import get_model
    old = os.environ.pop("CAMBRIAN_MODEL", None)
    try:
        model = get_model(0)
        assert model == "claude-sonnet-4-6"
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MODEL"] = old


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL (default claude-opus-4-6)."""
    import os
    from src.generate import get_model
    old = os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        model = get_model(1)
        assert model == "claude-opus-4-6"
        model2 = get_model(2)
        assert model2 == "claude-opus-4-6"
    finally:
        if old is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = old


def test_system_prompt_contains_rules() -> None:
    """System prompt contains required rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
    assert "Python 3.14" in SYSTEM_PROMPT
