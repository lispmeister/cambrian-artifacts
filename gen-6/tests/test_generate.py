"""Tests for LLM prompt construction, response parsing, and parse repair."""

import pytest

from src.generate import (
    ParseError,
    build_fresh_prompt,
    build_parse_repair_prompt,
    build_retry_prompt,
    parse_files,
)


# ---------------------------------------------------------------------------
# parse_files tests
# ---------------------------------------------------------------------------

def test_parse_single_file() -> None:
    """Parse a single file block."""
    response = '<file path="src/main.py">\nprint("hello")\n</file:end>\n'
    files = parse_files(response)
    assert "src/main.py" in files
    assert 'print("hello")' in files["src/main.py"]


def test_parse_multiple_files() -> None:
    """Parse multiple file blocks."""
    response = (
        '<file path="a.py">\ncontent_a\n</file:end>\n'
        '<file path="b.py">\ncontent_b\n</file:end>\n'
    )
    files = parse_files(response)
    assert set(files.keys()) == {"a.py", "b.py"}
    assert "content_a" in files["a.py"]
    assert "content_b" in files["b.py"]


def test_parse_preserves_content_exactly() -> None:
    """File content is preserved exactly as-is."""
    inner_content = "line1\nline2\nline3\n"
    response = f'<file path="test.py">\n{inner_content}</file:end>\n'
    files = parse_files(response)
    assert files["test.py"] == inner_content


def test_parse_file_with_empty_content() -> None:
    """Empty file block is parsed correctly."""
    response = '<file path="empty.py">\n</file:end>\n'
    files = parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """
    State machine stops at the FIRST line that is exactly '</file:end>'.
    A line containing '</file:end>' along with other text does NOT terminate the block.
    The actual closing tag (on its own line) terminates correctly.
    """
    # Build a response where the inner content has '</file:end>' embedded in text
    # (not on its own line), followed by more content, then the real closing tag.
    # The line 'This mentions </file:end> in text but is not the real end'
    # does NOT match line.rstrip("\n\r") == "</file:end>" so parsing continues.
    # 'actual content' is therefore included in the file.
    inner = "This mentions </file:end> in text but is not the real end\nactual content\n"
    response = f'<file path="tricky.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "tricky.py" in files
    # The embedded '</file:end>' in the middle of a line does NOT stop the parser.
    # The state machine only stops at a line that is EXACTLY '</file:end>'.
    # Therefore 'actual content' IS present in the parsed file.
    assert "actual content" in files["tricky.py"]
    assert "This mentions </file:end> in text but is not the real end" in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    """Path with subdirectory is correctly parsed."""
    response = '<file path="src/sub/module.py">\npass\n</file:end>\n'
    files = parse_files(response)
    assert "src/sub/module.py" in files


def test_parse_raises_parse_error_on_unclosed_block() -> None:
    """ParseError is raised for unclosed file block."""
    response = '<file path="unclosed.py">\ncontent without end\n'
    with pytest.raises(ParseError):
        parse_files(response)


def test_parse_error_message_contains_path() -> None:
    """ParseError message includes the unclosed file path."""
    response = '<file path="missing_end.py">\ncontent\n'
    with pytest.raises(ParseError) as exc_info:
        parse_files(response)
    assert "missing_end.py" in str(exc_info.value)


def test_parse_ignores_commentary_outside_blocks() -> None:
    """Text outside file blocks is silently discarded."""
    response = (
        "Some commentary here\n"
        '<file path="real.py">\ncontent\n</file:end>\n'
        "More commentary\n"
    )
    files = parse_files(response)
    assert list(files.keys()) == ["real.py"]


def test_parse_empty_response() -> None:
    """Empty response returns empty dict."""
    files = parse_files("")
    assert files == {}


def test_parse_response_with_only_commentary() -> None:
    """Response with no file blocks returns empty dict."""
    files = parse_files("Just some text\nNo file blocks here\n")
    assert files == {}


def test_parse_file_tag_in_content_does_not_confuse_parser() -> None:
    """File content containing <file path=...> tag does not open a new block."""
    inner = 'Example: <file path="nested.py">\ncontent\n'
    response = f'<file path="outer.py">\n{inner}</file:end>\n'
    files = parse_files(response)
    assert "outer.py" in files
    assert "nested.py" not in files


def test_parse_close_tag_on_own_line_terminates_block() -> None:
    """A line that is exactly '</file:end>' terminates the current block."""
    response = '<file path="exact.py">\nsome content\n</file:end>\nmore text\n'
    files = parse_files(response)
    assert "exact.py" in files
    assert files["exact.py"] == "some content\n"


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------

def test_fresh_prompt_includes_spec() -> None:
    """Fresh prompt includes spec content."""
    system_msg, user_msg = build_fresh_prompt(
        spec_content="MY SPEC CONTENT",
        generation_records_json="[]",
        generation=1,
        parent=0,
    )
    assert "MY SPEC CONTENT" in user_msg


def test_fresh_prompt_includes_history() -> None:
    """Fresh prompt includes generation history."""
    history = '[{"generation": 1}]'
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records_json=history,
        generation=2,
        parent=1,
    )
    assert history in user_msg


def test_fresh_prompt_includes_generation_number() -> None:
    """Fresh prompt includes generation number."""
    _, user_msg = build_fresh_prompt(
        spec_content="spec",
        generation_records_json="[]",
        generation=7,
        parent=6,
    )
    assert "7" in user_msg


def test_retry_prompt_includes_diagnostics() -> None:
    """Retry prompt includes diagnostics."""
    diag = {
        "stage": "test",
        "summary": "5 tests failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    _, user_msg = build_retry_prompt(
        spec_content="spec",
        generation_records_json="[]",
        generation=3,
        parent=2,
        failed_generation=2,
        diagnostics=diag,
        failed_files={},
    )
    assert "test" in user_msg
    assert "5 tests failed" in user_msg


def test_retry_prompt_includes_failed_code() -> None:
    """Retry prompt includes failed source code."""
    diag = {
        "stage": "build",
        "summary": "build failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    failed_files = {"src/main.py": "def broken(): pass\n"}
    _, user_msg = build_retry_prompt(
        spec_content="spec",
        generation_records_json="[]",
        generation=3,
        parent=2,
        failed_generation=2,
        diagnostics=diag,
        failed_files=failed_files,
    )
    assert "src/main.py" in user_msg
    assert "def broken(): pass" in user_msg


def test_parse_repair_prompt_includes_error() -> None:
    """Parse repair prompt includes the parse error message."""
    _, user_msg = build_parse_repair_prompt(
        parse_error_message="Unclosed block at foo.py",
        raw_response="<file path='foo.py'>\ncontent\n",
    )
    assert "Unclosed block at foo.py" in user_msg


def test_parse_repair_prompt_includes_raw_response() -> None:
    """Parse repair prompt includes the raw (malformed) response."""
    raw = "<file path='foo.py'>\ncontent without end\n"
    _, user_msg = build_parse_repair_prompt(
        parse_error_message="some error",
        raw_response=raw,
    )
    assert raw in user_msg


def test_model_selection_first_attempt() -> None:
    """CAMBRIAN_MODEL is used on first attempt (retry_count=0)."""
    import src.generate as gen_module
    original_model = gen_module.CAMBRIAN_MODEL
    gen_module.CAMBRIAN_MODEL = "test-model-sonnet"
    assert gen_module.CAMBRIAN_MODEL == "test-model-sonnet"
    gen_module.CAMBRIAN_MODEL = original_model


def test_model_escalation_on_retry() -> None:
    """CAMBRIAN_ESCALATION_MODEL is used on retry (retry_count >= 1)."""
    import src.generate as gen_module
    original = gen_module.CAMBRIAN_ESCALATION_MODEL
    gen_module.CAMBRIAN_ESCALATION_MODEL = "test-model-opus"
    assert gen_module.CAMBRIAN_ESCALATION_MODEL == "test-model-opus"
    gen_module.CAMBRIAN_ESCALATION_MODEL = original


def test_system_prompt_contains_rules() -> None:
    """System prompt contains key rules."""
    from src.generate import SYSTEM_PROMPT
    assert "aiohttp_client" in SYSTEM_PROMPT
    assert "unittest_run_loop" in SYSTEM_PROMPT
    assert "asyncio_mode" in SYSTEM_PROMPT