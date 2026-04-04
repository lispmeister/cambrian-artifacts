"""Tests for LLM integration: prompt building and response parsing."""


def test_parse_single_file() -> None:
    """Parse a single file block."""
    from src.generate import parse_files
    response = '<file path="hello.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert files == {"hello.py": 'print("hello")\n'}


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
    """File content is preserved exactly including trailing newlines."""
    from src.generate import parse_files
    content = "line1\nline2\nline3\n"
    response = f'<file path="x.py">\n{content}</file:end>\n'
    files = parse_files(response)
    assert files["x.py"] == content


def test_parse_file_with_empty_content() -> None:
    """Parse a file with empty content."""
    from src.generate import parse_files
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine: </file:end> embedded in a longer line is NOT a close tag.

    The line 'This mentions </file:end> in text but is not the real end'
    when stripped is NOT equal to '</file:end>' so it is accumulated as content.
    The actual close tag is the standalone '</file:end>' line at the end.
    Therefore ALL lines of 'inner' are in the file content.
    """
    from src.generate import parse_files
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # Both lines are preserved because neither is a standalone </file:end>
    assert "This mentions </file:end> in text but is not the real end\n" in files["tricky.py"]
    assert "actual content\n" in files["tricky.py"]
    assert files["tricky.py"] == inner


def test_parse_path_with_subdirectory() -> None:
    """Parse a file with a subdirectory in the path."""
    from src.generate import parse_files
    response = '<file path="src/module.py">\ncode\n</file:end>\n'
    files = parse_files(response)
    assert "src/module.py" in files
    assert files["src/module.py"] == "code\n"


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """Unclosed file block raises ParseError."""
    from src.generate import parse_files, ParseError
    response = '<file path="unclosed.py">\ncontent without end\n'
    try:
        parse_files(response)
        assert False, "Should have raised ParseError"
    except ParseError:
        pass


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed file path."""
    from src.generate import parse_files, ParseError
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
        assert False, "Should have raised ParseError"
    except ParseError as exc:
        assert "missing_end.py" in str(exc)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    from src.generate import parse_files
    response = (
        "Here is some commentary.\n"
        '<file path="a.py">\ncode\n</file:end>\n'
        "More commentary.\n"
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
    files = parse_files("Just some text with no file blocks.")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """A <file> open tag inside content does not start a new block."""
    from src.generate import parse_files
    response = (
        '<file path="outer.py">\n'
        'text with <file path="inner.py"> tag\n'
        '</file:end>\n'
    )
    files = parse_files(response)
    assert len(files) == 1
    assert "outer.py" in files
    # The entire line including "text with " prefix is preserved
    assert files["outer.py"] == 'text with <file path="inner.py"> tag\n'


# Spec test vectors from the spec document
def test_parse_spec_vector_1() -> None:
    """Spec Vector 1: </file:end> embedded in a longer line is NOT a close tag."""
    from src.generate import parse_files
    response = '<file path="a.py">\nx = "</file:end>"\n</file:end>\n'
    result = parse_files(response)
    assert result == {"a.py": 'x = "</file:end>"\n'}


def test_parse_spec_vector_2() -> None:
    """Spec Vector 2: Multiple files in sequence."""
    from src.generate import parse_files
    response = (
        '<file path="a.py">\nfoo\n</file:end>\n'
        '<file path="b.py">\nbar\n</file:end>\n'
    )
    result = parse_files(response)
    assert result == {"a.py": "foo\n", "b.py": "bar\n"}


def test_parse_spec_vector_3() -> None:
    """Spec Vector 3: </file:end> alone on its own line closes the block."""
    from src.generate import parse_files
    response = '<file path="a.py">\nline1\n</file:end>\ndiscarded\n'
    result = parse_files(response)
    assert result == {"a.py": "line1\n"}


def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    from src.generate import build_fresh_prompt
    messages = build_fresh_prompt("MY SPEC CONTENT", [], 1, 0)
    assert len(messages) == 1
    assert "MY SPEC CONTENT" in messages[0]["content"]


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    from src.generate import build_fresh_prompt
    history = [{"generation": 1, "outcome": "promoted"}]
    messages = build_fresh_prompt("spec", history, 2, 1)
    assert "promoted" in messages[0]["content"]


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes the generation number."""
    from src.generate import build_fresh_prompt
    messages = build_fresh_prompt("spec", [], 42, 41)
    assert "42" in messages[0]["content"]


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "test", "summary": "5 tests failed", "exit_code": 1}
    messages = build_retry_prompt("spec", [], 2, 1, diagnostics, {})
    content = messages[0]["content"]
    assert "test" in content
    assert "5 tests failed" in content


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    from src.generate import build_retry_prompt
    diagnostics = {"stage": "build", "summary": "error", "exit_code": 1}
    failed_files = {"src/main.py": "def broken(): pass\n"}
    messages = build_retry_prompt("spec", [], 2, 1, diagnostics, failed_files)
    assert "def broken(): pass" in messages[0]["content"]


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the error message."""
    from src.generate import build_repair_prompt
    messages = build_repair_prompt("Unclosed block at path='foo.py'", "raw response")
    assert "Unclosed block at path='foo.py'" in messages[0]["content"]


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw response."""
    from src.generate import build_repair_prompt
    messages = build_repair_prompt("some error", "THE RAW RESPONSE DATA")
    assert "THE RAW RESPONSE DATA" in messages[0]["content"]


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL env var."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_MODEL": "claude-test-model"}):
        model = get_model(0)
    assert model == "claude-test-model"


def test_model_escalation_on_retry() -> None:
    """Retry uses CAMBRIAN_ESCALATION_MODEL env var."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(os.environ, {"CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation"}):
        model = get_model(1)
    assert model == "claude-opus-escalation"


def test_model_default_first_attempt() -> None:
    """First attempt defaults to claude-sonnet-4-6."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    env = {k: v for k, v in os.environ.items()
           if k not in ("CAMBRIAN_MODEL", "CAMBRIAN_ESCALATION_MODEL")}
    with patch.dict(os.environ, env, clear=True):
        model = get_model(0)
    assert model == "claude-sonnet-4-6"


def test_model_default_retry() -> None:
    """Retry defaults to claude-opus-4-6."""
    import os
    from unittest.mock import patch
    from src.generate import get_model
    env = {k: v for k, v in os.environ.items()
           if k not in ("CAMBRIAN_MODEL", "CAMBRIAN_ESCALATION_MODEL")}
    with patch.dict(os.environ, env, clear=True):
        model = get_model(1)
    assert model == "claude-opus-4-6"


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "file path=" in SYSTEM_PROMPT
    assert "file:end" in SYSTEM_PROMPT
    assert "requirements.txt" in SYSTEM_PROMPT
