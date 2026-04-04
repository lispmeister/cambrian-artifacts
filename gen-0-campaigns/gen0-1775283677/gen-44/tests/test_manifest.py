"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash starts with sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("test spec content")
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        tmp_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches raw SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"test spec content for hashing"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert result == expected
    finally:
        tmp_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash does not include manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content a")
        (root / "manifest.json").write_bytes(b'{"cambrian-version": 1}')
        files_with = ["a.py", "manifest.json"]
        files_without = ["a.py"]
        hash_with = compute_artifact_hash(root, files_with)
        hash_without = compute_artifact_hash(root, files_without)
        assert hash_with == hash_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content a")
        hash1 = compute_artifact_hash(root, ["a.py"])
        (root / "a.py").write_bytes(b"content b")
        hash2 = compute_artifact_hash(root, ["a.py"])
        assert hash1 != hash2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash is computed in lexicographic sorted order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"aaa")
        (root / "b.py").write_bytes(b"bbb")
        hash1 = compute_artifact_hash(root, ["a.py", "b.py"])
        hash2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert hash1 == hash2  # order should not matter (sorted internally)


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Two files where path+content collision possible without separator
        (root / "ab.py").write_bytes(b"c")
        (root / "a.py").write_bytes(b"bc")
        hash1 = compute_artifact_hash(root, ["ab.py"])
        hash2 = compute_artifact_hash(root, ["a.py"])
        assert hash1 != hash2


def test_artifact_hash_format() -> None:
    """Artifact hash starts with sha256: prefix."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        result = compute_artifact_hash(root, ["a.py"])
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Manifest has all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
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
    """Manifest has correct generation numbers."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest token usage is correct."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 1234, "output": 5678},
        )
    assert manifest["token-usage"]["input"] == 1234
    assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
    assert manifest["entry"]["start"] == "python -m src.prime"
    assert "uv pip install" in manifest["entry"]["build"]
    assert "pytest" in manifest["entry"]["test"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        path = write_manifest(root, manifest)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [{"name": "health", "type": "http", "method": "GET", "path": "/health",
                  "expect": {"status": 200}}]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            contracts=contracts,
        )
    assert "contracts" in manifest
    assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in the manifest excludes manifest.json."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content a")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        expected_hash = compute_artifact_hash(root, ["a.py"])
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = (
        "Some spec content\n"
        "```contracts\n"
        '[{"name": "health", "type": "http"}]\n'
        "```\n"
    )
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "Some spec content without contracts block"
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None on invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```\n"
    result = extract_contracts_from_spec(spec)
    assert result is None
