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


def make_temp_artifact(files: dict[str, str]) -> tuple[Path, Path]:
    """Create a temporary artifact directory with given files. Returns (root, spec_path)."""
    tmpdir = Path(tempfile.mkdtemp())
    spec_path = tmpdir / "spec" / "CAMBRIAN-SPEC-005.md"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("# Test Spec\nContent here.", encoding="utf-8")

    for rel_path, content in files.items():
        dest = tmpdir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")

    return tmpdir, spec_path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"test content")
        path = Path(f.name)
    h = compute_spec_hash(path)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches manual SHA-256 computation."""
    content = b"test spec content"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert compute_spec_hash(path) == expected


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    root, _ = make_temp_artifact({"src/main.py": "print('hello')\n"})
    # With manifest.json
    (root / "manifest.json").write_text('{"test": true}', encoding="utf-8")
    files_with = ["src/main.py", "manifest.json"]
    files_without = ["src/main.py"]
    hash_with = compute_artifact_hash(root, files_with)
    hash_without = compute_artifact_hash(root, files_without)
    assert hash_with == hash_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    root, _ = make_temp_artifact({"src/main.py": "content_a\n"})
    hash_a = compute_artifact_hash(root, ["src/main.py"])
    (root / "src/main.py").write_text("content_b\n", encoding="utf-8")
    hash_b = compute_artifact_hash(root, ["src/main.py"])
    assert hash_a != hash_b


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash is independent of file list order."""
    root, _ = make_temp_artifact({
        "src/a.py": "aaa\n",
        "src/b.py": "bbb\n",
    })
    hash1 = compute_artifact_hash(root, ["src/a.py", "src/b.py"])
    hash2 = compute_artifact_hash(root, ["src/b.py", "src/a.py"])
    assert hash1 == hash2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    # Verify that path "ab" + content "c" != path "a" + content "bc"
    # by checking that the hash function uses the null separator
    root = Path(tempfile.mkdtemp())
    (root / "ab").write_bytes(b"c")
    (root / "a").write_bytes(b"bc")
    hash_ab = compute_artifact_hash(root, ["ab"])
    hash_a = compute_artifact_hash(root, ["a"])
    assert hash_ab != hash_a


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix."""
    root, _ = make_temp_artifact({"src/main.py": "content\n"})
    h = compute_artifact_hash(root, ["src/main.py"])
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Build manifest includes all MUST fields."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="claude-sonnet-4-6",
        token_usage={"input": 100, "output": 200},
    )
    required = [
        "cambrian-version", "generation", "parent-generation", "spec-hash",
        "artifact-hash", "producer-model", "token-usage", "files", "created-at", "entry",
    ]
    for field in required:
        assert field in manifest, f"Missing required field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent-generation."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=5,
        parent_generation=4,
        spec_path=spec_path,
        model="claude-sonnet-4-6",
        token_usage={"input": 0, "output": 0},
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest token-usage has input and output fields."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="test-model",
        token_usage={"input": 1234, "output": 5678},
    )
    assert manifest["token-usage"]["input"] == 1234
    assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="test-model",
        token_usage={"input": 0, "output": 0},
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points are present."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="test-model",
        token_usage={"input": 0, "output": 0},
    )
    assert "build" in manifest["entry"]
    assert "test" in manifest["entry"]
    assert "start" in manifest["entry"]
    assert "health" in manifest["entry"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    root = Path(tempfile.mkdtemp())
    manifest = {"cambrian-version": 1, "generation": 1}
    path = write_manifest(root, manifest)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["cambrian-version"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="test-model",
        token_usage={"input": 0, "output": 0},
    )
    assert "contracts" in manifest
    assert isinstance(manifest["contracts"], list)


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """artifact-hash in manifest excludes manifest.json."""
    root, spec_path = make_temp_artifact({"src/main.py": "content\n"})
    manifest = build_manifest(
        artifact_root=root,
        files=["src/main.py"],
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        model="test-model",
        token_usage={"input": 0, "output": 0},
    )
    # Compute expected hash without manifest
    files_no_manifest = [f for f in manifest["files"] if f != "manifest.json"]
    expected_hash = compute_artifact_hash(root, files_no_manifest)
    assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """Extracts contracts from spec with ```contracts block."""
    spec = '```contracts\n[{"name": "health", "type": "http"}]\n```'
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns None when no contracts block found."""
    spec = "# Spec\nNo contracts here."
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block has invalid JSON."""
    spec = '```contracts\nnot valid json\n```'
    result = extract_contracts_from_spec(spec)
    assert result is None
