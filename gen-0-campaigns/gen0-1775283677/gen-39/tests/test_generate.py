"""Tests for LLM prompt construction and response parsing."""
import pytest


def test_parse_single_file() -> None:
    """Parse a single file from LLM response."""
    from src.generate import parse_files
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    result = parse_files(response)
    assert result == {"hello.py": 'print("hello")\n'}


def test_parse_multiple_files() -> None:
    """Parse multiple files from LLM response."""
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
    """Parse a file with empty content."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    result = parse_files(response)
    assert "empty.py" in result
    assert result["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """State machine stops at the first </file:end> that is on its own line.

    The spec says:
    Vector 1: </file:end> embedded in a longer line — NOT a close tag.
    So 'This mentions </file:end> in text' does NOT close the block.
    But when </file:end> appears alone on its own line, it IS a close tag.
    """
    from src.generate import parse_files
    # Vector 1 from spec: </file:end> embedded in a longer line is NOT a close tag
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_path_with_subdirectory() -> None:
    """Parse a file with a subdirectory path."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncontent\n</file:end>\n'
    result = parse_files(response)
    assert "src/module.py" in result
    assert result["src/module.py"] == "content\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """ParseError is raised when a block is not closed."""
    from src.generate import parse_files, ParseError
    response = '<file path="unclosed.py">\ncontent without end\n'
    with pytest.raises(ParseError):
        parse_files(response)


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
        "Here is some commentary.\n"
        '<file path="code.py">\nprint(1)\n</file:end>\n'
        "More commentary.\n"
    )
    result = parse_files(response)
    assert list(result.keys()) == ["code.py"]
    assert result["code.py"] == "print(1)\n"


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    from src.generate import parse_files
    result = parse_files("")
    assert result == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with only commentary returns empty dict."""
    from src.generate import parse_files
    result = parse_files("This is just commentary, no files.")
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
    """Spec Vector 1: </file:end> in a longer line is NOT a close tag."""
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
    """Spec Vector 3: </file:end> alone closes block, content after is discarded."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    assert parse_files(response) == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("MY SPEC CONTENT", "[]", 1, 0)
    assert "MY SPEC CONTENT" in prompt


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", '[{"generation": 1}]', 2, 1)
    assert '[{"generation": 1}]' in prompt


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt("spec", "[]", 42, 41)
    assert "42" in prompt
    assert "41" in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "3 tests failed", "exit_code": 1}
    prompt = build_retry_prompt("spec", "[]", 2, 1, 1, diagnostics, {})
    assert "test" in prompt
    assert "3 tests failed" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "build", "summary": "build failed", "exit_code": 1}
    files = {"src/main.py": "def broken():\n    pass\n"}
    prompt = build_retry_prompt("spec", "[]", 2, 1, 1, diagnostics, files)
    assert "src/main.py" in prompt
    assert "def broken():" in prompt


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt("Unclosed block at path='foo.py'", "raw response")
    assert "Unclosed block at path='foo.py'" in prompt


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw response."""
    from src.generate import build_parse_repair_prompt
    raw = "some malformed LLM output"
    prompt = build_parse_repair_prompt("error msg", raw)
    assert raw in prompt


def test_model_selection_first_attempt() -> None:
    """First attempt uses the default model."""
    from src.generate import select_model, MODEL
    assert select_model(0) == MODEL


def test_model_escalation_on_retry() -> None:
    """Retry uses the escalation model."""
    from src.generate import select_model, ESCALATION_MODEL
    assert select_model(1) == ESCALATION_MODEL
    assert select_model(2) == ESCALATION_MODEL


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "code generator" in SYSTEM_PROMPT
    assert "<file path=" in SYSTEM_PROMPT
    assert "</file:end>" in SYSTEM_PROMPT
