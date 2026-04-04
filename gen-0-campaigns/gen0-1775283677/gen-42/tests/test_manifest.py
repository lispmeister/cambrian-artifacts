"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash starts with sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write("test spec content")
        spec_path = Path(f.name)
    try:
        h = compute_spec_hash(spec_path)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        spec_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"spec content for hashing"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        spec_path = Path(f.name)
    try:
        h = compute_spec_hash(spec_path)
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert h == expected
    finally:
        spec_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_bytes(b"code")
        (root / "manifest.json").write_bytes(b'{"test": true}')
        files = ["src/main.py", "manifest.json"]
        h = compute_artifact_hash(root, files)
        assert h.startswith("sha256:")
        # Without manifest
        h2 = compute_artifact_hash(root, ["src/main.py"])
        assert h == h2


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "file.py").write_bytes(b"version 1")
        h1 = compute_artifact_hash(root, ["file.py"])
        (root / "file.py").write_bytes(b"version 2")
        h2 = compute_artifact_hash(root, ["file.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"aaa")
        (root / "b.py").write_bytes(b"bbb")
        files_ab = ["a.py", "b.py"]
        files_ba = ["b.py", "a.py"]
        h1 = compute_artifact_hash(root, files_ab)
        h2 = compute_artifact_hash(root, files_ba)
        assert h1 == h2  # Same hash regardless of input order


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Test that null separator prevents collisions
        (root / "ab.py").write_bytes(b"c")
        (root / "a.py").write_bytes(b"bc")
        h1 = compute_artifact_hash(root, ["ab.py"])
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 != h2  # Different paths + contents should produce different hashes


def test_artifact_hash_format() -> None:
    """Artifact hash starts with sha256: and has 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"content")
        h = compute_artifact_hash(root, ["f.py"])
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest has all required MUST fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "__init__.py").write_bytes(b"")
        files = ["src/__init__.py"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=files,
            model="claude-test",
            token_usage={"input": 100, "output": 200},
        )
        required = [
            "cambrian-version", "generation", "parent-generation",
            "spec-hash", "artifact-hash", "producer-model",
            "token-usage", "files", "created-at", "entry",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"
        assert "build" in manifest["entry"]
        assert "test" in manifest["entry"]
        assert "start" in manifest["entry"]
        assert "health" in manifest["entry"]


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent-generation."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest includes token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 1234, "output": 5678},
        )
        assert manifest["token-usage"]["input"] == 1234
        assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest has cambrian-version: 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry.start uses module form."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["entry"]["start"] == "python -m src.prime"
        assert "pytest" in manifest["entry"]["test"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in artifact root."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        contracts = [{"name": "health", "type": "http", "method": "GET", "path": "/health",
                      "expect": {"status": 200}}]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-test",
            token_usage={"input": 0, "output": 0},
            contracts=contracts,
        )
        assert "contracts" in manifest
        assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """Artifact hash in manifest does not include manifest.json."""
    from src.manifest import build_manifest, write_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"content")
        files = ["f.py"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=files,
            model="test",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        # Recompute including manifest in file list
        files_with_manifest = ["f.py", "manifest.json"]
        h = compute_artifact_hash(root, files_with_manifest)
        # Both should equal same value (manifest.json excluded in both)
        assert manifest["artifact-hash"] == h


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts JSON block."""
    from src.manifest import extract_contracts_from_spec
    spec = """
Some spec content.

```contracts
[{"name": "health", "type": "http"}]
```

More content.
"""
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    result = extract_contracts_from_spec("No contracts here.")
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None
