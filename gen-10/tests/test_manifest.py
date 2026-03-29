"""Tests for manifest building and hash computation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from src.manifest import (
    build_manifest,
    compute_artifact_hash,
    compute_spec_hash,
    compute_spec_hash_from_content,
    extract_contracts_from_spec,
    write_manifest,
)


def test_spec_hash_format(tmp_path: Path) -> None:
    """Spec hash has sha256: prefix."""
    spec_file = tmp_path / "spec.md"
    spec_file.write_text("spec content", encoding="utf-8")
    hash_val = compute_spec_hash(spec_file)
    assert hash_val.startswith("sha256:")
    assert len(hash_val) == 7 + 64  # "sha256:" + 64 hex chars


def test_spec_hash_matches_sha256(tmp_path: Path) -> None:
    """Spec hash matches manual SHA-256 computation."""
    content = b"my spec content"
    spec_file = tmp_path / "spec.md"
    spec_file.write_bytes(content)
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert compute_spec_hash(spec_file) == expected


def test_artifact_hash_excludes_manifest(tmp_path: Path) -> None:
    """Artifact hash excludes manifest.json."""
    (tmp_path / "a.py").write_text("content a", encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"key": "value"}', encoding="utf-8")
    files_with_manifest = ["a.py", "manifest.json"]
    files_without_manifest = ["a.py"]
    hash_with = compute_artifact_hash(tmp_path, files_with_manifest)
    hash_without = compute_artifact_hash(tmp_path, files_without_manifest)
    assert hash_with == hash_without


def test_artifact_hash_includes_file_content(tmp_path: Path) -> None:
    """Artifact hash changes when file content changes."""
    f = tmp_path / "file.py"
    f.write_text("version 1", encoding="utf-8")
    hash1 = compute_artifact_hash(tmp_path, ["file.py"])
    f.write_text("version 2", encoding="utf-8")
    hash2 = compute_artifact_hash(tmp_path, ["file.py"])
    assert hash1 != hash2


def test_artifact_hash_uses_sorted_order(tmp_path: Path) -> None:
    """Artifact hash is the same regardless of file list order."""
    (tmp_path / "a.py").write_text("aaa", encoding="utf-8")
    (tmp_path / "b.py").write_text("bbb", encoding="utf-8")
    hash1 = compute_artifact_hash(tmp_path, ["a.py", "b.py"])
    hash2 = compute_artifact_hash(tmp_path, ["b.py", "a.py"])
    assert hash1 == hash2


def test_artifact_hash_has_null_separator(tmp_path: Path) -> None:
    """
    Artifact hash uses null separator between path and content.
    Verify by checking that path collision does not produce same hash.
    """
    # file "ab" with content "c" vs file "a" with content "bc"
    # Without null separator: both produce hash("ab" + "c") == hash("a" + "bc")
    # With null separator: hash("ab\0c") != hash("a\0bc")
    f1 = tmp_path / "ab"
    f1.write_bytes(b"c")
    hash1 = compute_artifact_hash(tmp_path, ["ab"])

    f2 = tmp_path / "a"
    f2.write_bytes(b"bc")
    hash2 = compute_artifact_hash(tmp_path, ["a"])

    # They should be different due to null separator
    assert hash1 != hash2


def test_artifact_hash_format(tmp_path: Path) -> None:
    """Artifact hash has sha256: prefix and 64 hex chars."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    h = compute_artifact_hash(tmp_path, ["f.py"])
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_build_manifest_has_required_fields(tmp_path: Path) -> None:
    """Built manifest contains all MUST fields."""
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="claude-test",
        token_usage={"input": 100, "output": 200},
        files=["src/prime.py"],
    )
    required = [
        "cambrian-version", "generation", "parent-generation",
        "spec-hash", "artifact-hash", "producer-model",
        "token-usage", "files", "created_at", "entry",
    ]
    for field in required:
        assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest generation and parent-generation are set correctly."""
    manifest = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest token-usage has input and output fields."""
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 1234, "output": 5678},
        files=[],
    )
    assert manifest["token-usage"]["input"] == 1234
    assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry has build, test, start, health fields."""
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    entry = manifest["entry"]
    assert "build" in entry
    assert "test" in entry
    assert "start" in entry
    assert "health" in entry


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    """write_manifest creates manifest.json in the artifact root."""
    manifest = {"cambrian-version": 1, "generation": 1}
    result = write_manifest(tmp_path, manifest)
    assert result == tmp_path / "manifest.json"
    assert (tmp_path / "manifest.json").exists()
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["cambrian-version"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when spec has contracts block."""
    spec_with_contracts = '''
Some spec content.

```contracts
[{"name": "health", "type": "http", "method": "GET", "path": "/health",
  "expect": {"status": 200, "body": {"ok": true}}}]
```

More content.
'''
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
        spec_content=spec_with_contracts,
    )
    assert "contracts" in manifest
    assert isinstance(manifest["contracts"], list)
    assert manifest["contracts"][0]["name"] == "health"


def test_artifact_hash_in_manifest_excludes_manifest(tmp_path: Path) -> None:
    """The artifact-hash in a manifest excludes manifest.json itself."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("code", encoding="utf-8")
    files = ["src/main.py", "manifest.json"]
    hash_val = compute_artifact_hash(tmp_path, files)
    # Should be same as without manifest.json
    hash_val2 = compute_artifact_hash(tmp_path, ["src/main.py"])
    assert hash_val == hash_val2


def test_extract_contracts_from_spec_found() -> None:
    """Contracts are extracted from spec when present."""
    spec = '```contracts\n[{"name": "test"}]\n```'
    result = extract_contracts_from_spec(spec)
    assert result == [{"name": "test"}]


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns None when no contracts block present."""
    result = extract_contracts_from_spec("No contracts here.")
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block contains invalid JSON."""
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None