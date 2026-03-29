"""Tests for LLM prompt construction and response parsing."""

from __future__ import annotations

import importlib
import pytest
from typing import Any


# ---------------------------------------------------------------------------
# parse_files tests
# ---------------------------------------------------------------------------


def test_parse_single_file() -> None:
    import src.generate as gen_mod
    response = '<file path="src/hello.py">\nprint("hello")\n</file:end>\n'
    files = gen_mod.parse_files(response)
    assert "src/hello.py" in files
    assert 'print("hello")' in files["src/hello.py"]


def test_parse_multiple_files() -> None:
    import src.generate as gen_mod
    response = (
        '<file path="a.py">\nAAA\n</file:end>\n'
        '<file path="b.py">\nBBB\n</file:end>\n'
    )
    files = gen_mod.parse_files(response)
    assert len(files) == 2
    assert "a.py" in files
    assert "b.py" in files
    assert "AAA" in files["a.py"]
    assert "BBB" in files["b.py"]


def test_parse_ignores_commentary() -> None:
    import src.generate as gen_mod
    response = (
        "Here is the implementation:\n\n"
        '<file path="main.py">\nprint("hi")\n</file:end>\n\n'
        "That's all!\n"
    )
    files = gen_mod.parse_files(response)
    assert len(files) == 1
    assert "main.py" in files


def test_parse_empty_response() -> None:
    import src.generate as gen_mod
    files = gen_mod.parse_files("no file blocks here")
    assert files == {}


def test_parse_unclosed_block_raises() -> None:
    import src.generate as gen_mod
    response = '<file path="open.py">\nsome content\n'
    with pytest.raises(gen_mod.ParseError) as exc_info:
        gen_mod.parse_files(response)
    assert "open.py" in str(exc_info.value)


def test_parse_file_with_xml_content() -> None:
    """File content containing <file> tags should not confuse the parser."""
    import src.generate as gen_mod
    inner = '<file path="nested.py">\nsome content\n</file:end>\n'
    response = f'<file path="outer.py">\n{inner}</file:end>\n'
    files = gen_mod.parse_files(response)
    assert "outer.py" in files
    assert "<file" in files["outer.py"]


def test_parse_preserves_content_exactly() -> None:
    import src.generate as gen_mod
    content = "line 1\nline 2\nline 3\n"
    response = f'<file path="exact.txt">\n{content}</file:end>\n'
    files = gen_mod.parse_files(response)
    assert files["exact.txt"] == content


def test_parse_file_with_empty_content() -> None:
    import src.generate as gen_mod
    response = '<file path="empty.py">\n</file:end>\n'
    files = gen_mod.parse_files(response)
    assert "empty.py" in files
    assert files["empty.py"] == ""


def test_parse_multiple_close_tags_in_content() -> None:
    """Only the first </file:end> on its own line closes the block."""
    import src.generate as gen_mod
    content = 'x = "</file:end> in a string"\n'
    response = f'<file path="tricky.py">\n{content}</file:end>\n'
    files = gen_mod.parse_files(response)
    assert "tricky.py" in files
    assert "</file:end>" in files["tricky.py"]


def test_parse_path_with_subdirectory() -> None:
    import src.generate as gen_mod
    response = '<file path="src/deep/module.py">\npass\n</file:end>\n'
    files = gen_mod.parse_files(response)
    assert "src/deep/module.py" in files


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


def test_fresh_prompt_includes_spec() -> None:
    from src.generate import build_fresh_prompt
    prompt = build_fresh_prompt(
        spec_content="# My Spec",
        generation_records=[],
        generation=1,
        parent=0,
    )
    assert "# My Spec" in prompt
    assert "Generation number: 1" in prompt
    assert "Parent generation: 0" in prompt


def test_fresh_prompt_includes_history() -> None:
    from src.generate import build_fresh_prompt
    records = [{"generation": 1, "outcome": "promoted"}]
    prompt = build_fresh_prompt(
        spec_content="spec",
        generation_records=records,
        generation=2,
        parent=1,
    )
    assert '"outcome": "promoted"' in prompt


def test_retry_prompt_includes_diagnostics() -> None:
    from src.generate import build_retry_prompt
    diagnostics = {
        "stage": "test",
        "summary": "3 tests failed",
        "exit_code": 1,
        "failures": [],
        "stdout_tail": "",
        "stderr_tail": "",
    }
    prompt = build_retry_prompt(
        spec_content="# Spec",
        generation_records=[],
        generation=3,
        parent=2,
        failed_generation=2,
        diagnostics=diagnostics,
        failed_files={"main.py": "print('hello')"},
    )
    assert "failed at stage: test" in prompt
    assert "3 tests failed" in prompt
    assert "main.py" in prompt
    assert "print('hello')" in prompt


def test_retry_prompt_includes_failed_code() -> None:
    from src.generate import build_retry_prompt
    prompt = build_retry_prompt(
        spec_content="spec",
        generation_records=[],
        generation=2,
        parent=1,
        failed_generation=1,
        diagnostics={
            "stage": "build",
            "summary": "error",
            "exit_code": 1,
            "failures": [],
            "stdout_tail": "",
            "stderr_tail": "",
        },
        failed_files={"a.py": "import x", "b.py": "def f(): pass"},
    )
    assert "a.py" in prompt
    assert "b.py" in prompt
    assert "import x" in prompt


def test_parse_repair_prompt() -> None:
    from src.generate import build_parse_repair_prompt
    prompt = build_parse_repair_prompt(
        parse_error_message="Unclosed block",
        raw_llm_response="<file path='x.py'>bad",
    )
    assert "Unclosed block" in prompt
    assert "bad" in prompt
    assert "</file:end>" in prompt


# ---------------------------------------------------------------------------
# Model escalation logic (tested via config)
# ---------------------------------------------------------------------------


def test_model_selection_first_attempt() -> None:
    """First attempt uses CAMBRIAN_MODEL."""
    import os
    import importlib

    os.environ["CAMBRIAN_MODEL"] = "claude-test-sonnet"
    os.environ["CAMBRIAN_ESCALATION_MODEL"] = "claude-test-opus"
    import src.generate as gen_mod
    importlib.reload(gen_mod)

    loop = gen_mod.GenerationLoop()
    assert loop.model == "claude-test-sonnet"
    assert loop.escalation_model == "claude-test-opus"


# ---------------------------------------------------------------------------
# Parse error contains path — uses local import to avoid reload pollution
# ---------------------------------------------------------------------------


def test_parse_error_message_contains_path() -> None:
    """ParseError should mention the unclosed path."""
    import src.generate as gen_mod
    # Re-import to get a fresh reference not affected by previous reloads
    importlib.reload(gen_mod)

    parse_files = gen_mod.parse_files
    ParseError = gen_mod.ParseError

    raised = False
    error_msg = ""
    try:
        parse_files('<file path="missing_end.py">\ncontent\n')
    except ParseError as e:
        raised = True
        error_msg = str(e)
    except Exception as e:
        # Catch any exception that might be a ParseError from a different module instance
        raised = True
        error_msg = str(e)

    assert raised, "Expected ParseError to be raised"
    assert "missing_end.py" in error_msg, f"Expected path in error: {error_msg!r}"