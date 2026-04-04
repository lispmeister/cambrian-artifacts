"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"# Test Spec\n")
        path = Path(f.name)
    try:
        h = compute_spec_hash(path)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches manual SHA-256 computation."""
    from src.manifest import compute_spec_hash
    content = b"# Test Spec Content\nSome content here.\n"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        h = compute_spec_hash(path)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert h == expected
    finally:
        path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content_a", encoding="utf-8")
        (root / "manifest.json").write_text('{"key": "value"}', encoding="utf-8")

        files_with = ["a.py", "manifest.json"]
        files_without = ["a.py"]

        h_with = compute_artifact_hash(root, files_with)
        h_without = compute_artifact_hash(root, files_without)
        assert h_with == h_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content_a", encoding="utf-8")
        h1 = compute_artifact_hash(root, ["a.py"])

        (root / "a.py").write_text("content_b", encoding="utf-8")
        h2 = compute_artifact_hash(root, ["a.py"])

        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort of file paths."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_text("b_content", encoding="utf-8")
        (root / "a.py").write_text("a_content", encoding="utf-8")

        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash includes null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Manual computation
        (root / "test.py").write_text("content", encoding="utf-8")

        hasher = hashlib.sha256()
        path_bytes = b"test.py"
        content_bytes = b"content"
        hasher.update(path_bytes)
        hasher.update(b"\0")
        hasher.update(content_bytes)
        expected = f"sha256:{hasher.hexdigest()}"

        result = compute_artifact_hash(root, ["test.py"])
        assert result == expected


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix and 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")
        h = compute_artifact_hash(root, ["a.py"])
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest has all required MUST fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("# main", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["src/main.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 100, "output": 200},
        )

        required_fields = [
            "cambrian-version", "generation", "parent-generation",
            "spec-hash", "artifact-hash", "producer-model",
            "token-usage", "files", "created-at", "entry",
        ]
        for field in required_fields:
            assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Built manifest has correct generation and parent numbers."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            producer_model="claude-test",
            token_usage={"input": 50, "output": 100},
        )

        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Built manifest includes correct token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            producer_model="claude-test",
            token_usage={"input": 1234, "output": 5678},
        )

        assert manifest["token-usage"] == {"input": 1234, "output": 5678}


def test_build_manifest_cambrian_version() -> None:
    """Built manifest has cambrian-version = 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
        )

        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Built manifest has correct entry points including module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
        )

        entry = manifest["entry"]
        assert "build" in entry
        assert "test" in entry
        assert "start" in entry
        assert "health" in entry
        # entry.start MUST use module form
        assert "python -m src.prime" == entry["start"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in the artifact root."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
        )
        write_manifest(root, manifest)

        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Build manifest includes contracts."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "g" * 64,
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
        )

        assert "contracts" in manifest
        assert isinstance(manifest["contracts"], list)
        assert len(manifest["contracts"]) > 0


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in the manifest excludes manifest.json."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content", encoding="utf-8")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "h" * 64,
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
        )

        expected_hash = compute_artifact_hash(root, ["a.py"])
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds JSON array in contracts block."""
    from src.manifest import extract_contracts_from_spec
    contracts = [{"name": "health", "type": "http"}]
    spec = f'Some text\n```contracts\n{json.dumps(contracts)}\n```\nMore text'
    result = extract_contracts_from_spec(spec)
    assert result == contracts


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "# No contracts here\nJust regular content."
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None
