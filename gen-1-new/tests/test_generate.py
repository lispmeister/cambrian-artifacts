"""Tests for LLM prompt construction and response parsing."""
from __future__ import annotations

import pytest
from src.generate import parse_files, build_fresh_prompt, build_retry_prompt


class TestParseFiles:
    def test_single_file(self):
        response = '<file path="src/main.py">print("hello")</file:end>'
        files = parse_files(response)
        assert files == {"src/main.py": 'print("hello")'}

    def test_multiple_files(self):
        response = (
            '<file path="src/a.py">a = 1</file:end>\n'
            '<file path="src/b.py">b = 2</file:end>'
        )
        files = parse_files(response)
        assert len(files) == 2
        assert files["src/a.py"] == "a = 1"
        assert files["src/b.py"] == "b = 2"

    def test_strips_surrounding_newlines(self):
        response = '<file path="x.py">\nline1\nline2\n</file:end>'
        files = parse_files(response)
        assert files["x.py"] == "line1\nline2"

    def test_ignores_text_outside_blocks(self):
        response = (
            "Here is the code:\n\n"
            '<file path="x.py">code</file:end>\n\n'
            "That's all."
        )
        files = parse_files(response)
        assert list(files.keys()) == ["x.py"]

    def test_empty_response(self):
        assert parse_files("") == {}

    def test_malformed_response(self):
        assert parse_files("no file blocks here") == {}

    def test_multiline_content(self):
        response = (
            '<file path="main.py">def main():\n    pass\n\n'
            'if __name__ == "__main__":\n    main()</file:end>'
        )
        files = parse_files(response)
        assert "def main():" in files["main.py"]
        assert "if __name__" in files["main.py"]

    def test_normalizes_crlf_to_lf(self):
        response = '<file path="x.py">line1\r\nline2\r\nline3</file:end>'
        files = parse_files(response)
        assert files["x.py"] == "line1\nline2\nline3"
        assert "\r" not in files["x.py"]

    def test_handles_mixed_line_endings(self):
        response = '<file path="x.py">line1\r\nline2\nline3\r\n</file:end>'
        files = parse_files(response)
        assert files["x.py"] == "line1\nline2\nline3"

    def test_nested_path_separators(self):
        response = '<file path="src/sub/module.py">x = 1</file:end>'
        files = parse_files(response)
        assert "src/sub/module.py" in files

    def test_file_with_xml_like_content(self):
        response = '<file path="x.py">result = "<div>"</file:end>'
        files = parse_files(response)
        assert files["x.py"] == 'result = "<div>"'

    def test_nested_file_tags_in_content(self):
        """Content containing inner <file>...</file> tags must not confuse the parser."""
        response = (
            '<file path="tests/test_gen.py">\n'
            'def test_single():\n'
            '    response = \'<file path="x.py">code</file>\'\n'
            '    assert parse_files(response) == {"x.py": "code"}\n'
            '</file:end>'
        )
        files = parse_files(response)
        assert "tests/test_gen.py" in files
        assert '</file>' in files["tests/test_gen.py"]


class TestBuildFreshPrompt:
    def test_includes_spec(self):
        prompt = build_fresh_prompt("# My Spec", [], 1, 0)
        assert "# My Spec" in prompt

    def test_includes_generation_number(self):
        prompt = build_fresh_prompt("spec", [], 5, 3)
        assert "Generation number: 5" in prompt
        assert "Parent generation: 3" in prompt

    def test_includes_history(self):
        records = [{"generation": 1, "outcome": "promoted"}]
        prompt = build_fresh_prompt("spec", records, 2, 1)
        assert "promoted" in prompt

    def test_empty_history_renders_empty_array(self):
        prompt = build_fresh_prompt("spec", [], 1, 0)
        assert "[]" in prompt

    def test_returns_string(self):
        prompt = build_fresh_prompt("spec", [], 1, 0)
        assert isinstance(prompt, str)


class TestBuildRetryPrompt:
    def test_includes_diagnostics(self):
        prompt = build_retry_prompt(
            spec_content="spec",
            generation_records=[],
            generation=3,
            parent=0,
            failed_generation=2,
            diagnostics={"stage": "test", "summary": "5 tests failed"},
            failed_files={"src/main.py": "broken code"},
        )
        assert "5 tests failed" in prompt
        assert "broken code" in prompt
        assert "Generation 2 failed" in prompt

    def test_includes_spec_content(self):
        prompt = build_retry_prompt(
            spec_content="# Genome Spec",
            generation_records=[],
            generation=2,
            parent=0,
            failed_generation=1,
            diagnostics={"stage": "build", "summary": "pip error"},
            failed_files={},
        )
        assert "# Genome Spec" in prompt

    def test_includes_stage(self):
        prompt = build_retry_prompt(
            spec_content="spec",
            generation_records=[],
            generation=2,
            parent=0,
            failed_generation=1,
            diagnostics={"stage": "build", "summary": "error"},
            failed_files={},
        )
        assert "build" in prompt

    def test_handles_empty_failed_files(self):
        prompt = build_retry_prompt(
            spec_content="spec",
            generation_records=[],
            generation=2,
            parent=0,
            failed_generation=1,
            diagnostics={"stage": "test", "summary": "error"},
            failed_files={},
        )
        assert "no source code available" in prompt

    def test_includes_generation_numbers(self):
        prompt = build_retry_prompt(
            spec_content="spec",
            generation_records=[],
            generation=4,
            parent=2,
            failed_generation=3,
            diagnostics={"stage": "health", "summary": "timeout"},
            failed_files={},
        )
        assert "Generation number: 4" in prompt
        assert "Parent generation: 2" in prompt
