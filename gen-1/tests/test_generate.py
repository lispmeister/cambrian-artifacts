"""Tests for LLM prompt construction and response parsing."""
import pytest
from src.generate import parse_files, build_fresh_prompt, build_retry_prompt


class TestParseFiles:
    def test_single_file(self):
        response = '<file path="src/main.py">print("hello")</file>'
        files = parse_files(response)
        assert files == {"src/main.py": 'print("hello")'}

    def test_multiple_files(self):
        response = (
            '<file path="src/a.py">a = 1</file>\n'
            '<file path="src/b.py">b = 2</file>'
        )
        files = parse_files(response)
        assert len(files) == 2
        assert files["src/a.py"] == "a = 1"
        assert files["src/b.py"] == "b = 2"

    def test_strips_surrounding_newlines(self):
        response = '<file path="x.py">\nline1\nline2\n</file>'
        files = parse_files(response)
        assert files["x.py"] == "line1\nline2"

    def test_ignores_text_outside_blocks(self):
        response = (
            "Here is the code:\n\n"
            '<file path="x.py">code</file>\n\n'
            "That's all."
        )
        files = parse_files(response)
        assert list(files.keys()) == ["x.py"]

    def test_empty_response(self):
        assert parse_files("") == {}

    def test_malformed_response(self):
        assert parse_files("no file blocks here") == {}

    def test_multiline_content(self):
        response = '<file path="main.py">def main():\n    pass\n\nif __name__ == "__main__":\n    main()</file>'
        files = parse_files(response)
        assert "def main():" in files["main.py"]
        assert "if __name__" in files["main.py"]


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
