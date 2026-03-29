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
    extract_contracts_from_spec,
    write_manifest,
)


def test_spec_hash_format(tmp_path: Path) -> None:
    """spec-hash has sha256: prefix."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec content", encoding="utf-8")
    h = compute_spec_hash(spec)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


def test_spec_hash_matches_sha256(tmp_path: Path) -> None:
    """spec-hash matches manual SHA-256 computation."""
    content = b"# Spec content for testing"
    spec = tmp_path / "spec.md"
    spec.write_bytes(content)
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert compute_spec_hash(spec) == expected


def test_artifact_hash_excludes_manifest(tmp_path: Path) -> None:
    """artifact-hash excludes manifest.json."""
    (tmp_path / "a.py").write_text("content a", encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"key": "value"}', encoding="utf-8")

    files_with = ["a.py", "manifest.json"]
    files_without = ["a.py"]

    hash_with = compute_artifact_hash(tmp_path, files_with)
    hash_without = compute_artifact_hash(tmp_path, files_without)
    assert hash_with == hash_without


def test_artifact_hash_includes_file_content(tmp_path: Path) -> None:
    """artifact-hash changes when file content changes."""
    f = tmp_path / "a.py"
    f.write_text("content v1", encoding="utf-8")
    h1 = compute_artifact_hash(tmp_path, ["a.py"])

    f.write_text("content v2", encoding="utf-8")
    h2 = compute_artifact_hash(tmp_path, ["a.py"])

    assert h1 != h2


def test_artifact_hash_uses_sorted_order(tmp_path: Path) -> None:
    """artifact-hash is independent of file list order (uses sorted)."""
    (tmp_path / "a.py").write_text("content a", encoding="utf-8")
    (tmp_path / "b.py").write_text("content b", encoding="utf-8")

    h1 = compute_artifact_hash(tmp_path, ["a.py", "b.py"])
    h2 = compute_artifact_hash(tmp_path, ["b.py", "a.py"])
    assert h1 == h2


def test_artifact_hash_has_null_separator(tmp_path: Path) -> None:
    """Null separator prevents hash collisions."""
    # Create two files that would collide without separator
    # path1 + content1 == path2 + content2 boundary collision
    (tmp_path / "ab.py").write_text("c", encoding="utf-8")
    (tmp_path / "a.py").write_text("bc", encoding="utf-8")

    h1 = compute_artifact_hash(tmp_path, ["ab.py"])
    h2 = compute_artifact_hash(tmp_path, ["a.py"])
    assert h1 != h2


def test_artifact_hash_format(tmp_path: Path) -> None:
    """artifact-hash has sha256: prefix and correct length."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    h = compute_artifact_hash(tmp_path, ["f.py"])
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_build_manifest_has_required_fields(tmp_path: Path) -> None:
    """Built manifest has all MUST fields."""
    (tmp_path / "src.py").write_text("code", encoding="utf-8")
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_root=tmp_path,
        files=["src.py"],
        model="claude-sonnet-4-6",
        token_usage={"input": 100, "output": 200},
    )
    required = [
        "cambrian-version", "generation", "parent-generation",
        "spec-hash", "artifact-hash", "producer-model",
        "token-usage", "files", "created_at", "entry",
    ]
    for field in required:
        assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers(tmp_path: Path) -> None:
    """Manifest has correct generation numbers."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    manifest = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "b" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage(tmp_path: Path) -> None:
    """Manifest token-usage has input and output."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "c" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 12345, "output": 67890},
    )
    assert manifest["token-usage"]["input"] == 12345
    assert manifest["token-usage"]["output"] == 67890


def test_build_manifest_cambrian_version(tmp_path: Path) -> None:
    """Manifest cambrian-version is 1."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "d" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points(tmp_path: Path) -> None:
    """Manifest entry points are present."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "e" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
    )
    entry = manifest["entry"]
    assert "build" in entry
    assert "test" in entry
    assert "start" in entry
    assert "health" in entry


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    """write_manifest creates manifest.json."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "f" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
    )
    path = write_manifest(tmp_path, manifest)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["generation"] == 1


def test_build_manifest_with_contracts(tmp_path: Path) -> None:
    """Manifest includes contracts when provided."""
    (tmp_path / "f.py").write_text("x", encoding="utf-8")
    contracts = [
        {"name": "health", "type": "http", "method": "GET", "path": "/health",
         "expect": {"status": 200}},
    ]
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
        contracts=contracts,
    )
    assert "contracts" in manifest
    assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest(tmp_path: Path) -> None:
    """The artifact-hash in the manifest excludes manifest.json itself."""
    (tmp_path / "src.py").write_text("code here", encoding="utf-8")

    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_root=tmp_path,
        files=["src.py"],
        model="model",
        token_usage={"input": 0, "output": 0},
    )

    # Compute expected hash manually (without manifest.json)
    expected = compute_artifact_hash(tmp_path, ["src.py", "manifest.json"])
    assert manifest["artifact-hash"] == expected


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts in spec."""
    spec = """
Some content

```contracts
[
  {"name": "health", "type": "http", "method": "GET", "path": "/health",
   "expect": {"status": 200, "body": {"ok": true}}}
]
```

More content
"""
    contracts = extract_contracts_from_spec(spec)
    assert contracts is not None
    assert len(contracts) == 1
    assert contracts[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    spec = "# Spec\n\nNo contracts here.\n"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    spec = "```contracts\nnot valid json\n```\n"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None