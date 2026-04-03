"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest


def test_spec_hash_format() -> None:
    """Spec hash has correct sha256: prefix format."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("test spec content")
        f.flush()
        path = Path(f.name)
    try:
        result = compute_spec_hash(path)
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64
    finally:
        path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches manual SHA-256 computation."""
    from src.manifest import compute_spec_hash
    content = b"test spec content for hashing"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        result = compute_spec_hash(path)
        expected = f"sha256:{hashlib.sha256(content).hexdigest()}"
        assert result == expected
    finally:
        path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content a")
        (root / "manifest.json").write_text('{"key": "value"}')
        files = ["a.py", "manifest.json"]

        result_with = compute_artifact_hash(root, files)

        files_without = ["a.py"]
        result_without = compute_artifact_hash(root, files_without)

        assert result_with == result_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("original content")
        h1 = compute_artifact_hash(root, ["a.py"])

        (root / "a.py").write_text("modified content")
        h2 = compute_artifact_hash(root, ["a.py"])

        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "z.py").write_text("z content")
        (root / "a.py").write_text("a content")

        h1 = compute_artifact_hash(root, ["z.py", "a.py"])
        h2 = compute_artifact_hash(root, ["a.py", "z.py"])

        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # These two files would collide without null separator:
        # "a" + "b" == "ab" + ""  (without separator)
        (root / "a").write_bytes(b"b")
        h1 = compute_artifact_hash(root, ["a"])

        # Verify by manual computation with null separator
        hasher = hashlib.sha256()
        hasher.update(b"a")
        hasher.update(b"\0")
        hasher.update(b"b")
        expected = f"sha256:{hasher.hexdigest()}"

        assert h1 == expected


def test_artifact_hash_format() -> None:
    """Artifact hash has correct sha256: prefix format."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content")
        result = compute_artifact_hash(root, ["a.py"])
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest has all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec content")
        (root / "src").mkdir()
        (root / "src" / "__init__.py").write_text("")

        files = ["src/__init__.py", "manifest.json"]

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=files,
            model="claude-sonnet-4-6",
            token_usage={"input": 100, "output": 200},
            spec_content="spec content",
        )

        assert "cambrian-version" in manifest
        assert "generation" in manifest
        assert "parent-generation" in manifest
        assert "spec-hash" in manifest
        assert "artifact-hash" in manifest
        assert "producer-model" in manifest
        assert "token-usage" in manifest
        assert "files" in manifest
        assert "created-at" in manifest
        assert "entry" in manifest


def test_build_manifest_generation_numbers() -> None:
    """Manifest generation numbers are set correctly."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec")
        (root / "a.py").write_text("content")

        manifest = build_manifest(
            generation=5,
            parent_generation=4,
            spec_path=spec,
            artifact_root=root,
            files=["a.py", "manifest.json"],
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
            spec_content="spec",
        )

        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest token usage is set correctly."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec")
        (root / "a.py").write_text("content")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=["a.py", "manifest.json"],
            model="test-model",
            token_usage={"input": 1234, "output": 5678},
            spec_content="spec",
        )

        assert manifest["token-usage"]["input"] == 1234
        assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec")
        (root / "a.py").write_text("content")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=["a.py", "manifest.json"],
            model="test-model",
            token_usage={"input": 0, "output": 0},
            spec_content="spec",
        )

        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec")
        (root / "a.py").write_text("content")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=["a.py", "manifest.json"],
            model="test-model",
            token_usage={"input": 0, "output": 0},
            spec_content="spec",
        )

        assert manifest["entry"]["start"] == "python -m src.prime"
        assert "python -m pytest" in manifest["entry"]["test"]
        assert manifest["entry"]["health"] == "http://localhost:8401/health"


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json file."""
    from src.manifest import write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = {"test": "data", "cambrian-version": 1}
        write_manifest(root, manifest)
        assert (root / "manifest.json").exists()
        loaded = json.loads((root / "manifest.json").read_text())
        assert loaded["test"] == "data"


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts from spec when present."""
    from src.manifest import build_manifest
    spec_content = """Some spec text

```contracts
[{"name": "test-contract", "type": "http", "method": "GET", "path": "/test"}]
```

More text
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text(spec_content)
        (root / "a.py").write_text("content")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=["a.py", "manifest.json"],
            model="test-model",
            token_usage={"input": 0, "output": 0},
            spec_content=spec_content,
        )

        assert "contracts" in manifest
        assert len(manifest["contracts"]) == 1
        assert manifest["contracts"][0]["name"] == "test-contract"


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in the manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = root / "spec.md"
        spec.write_text("spec content")
        (root / "a.py").write_text("test content")

        files = ["a.py", "manifest.json"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_path=spec,
            artifact_root=root,
            files=files,
            model="test-model",
            token_usage={"input": 0, "output": 0},
            spec_content="spec content",
        )

        expected_hash = compute_artifact_hash(root, files)
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = """Some text

```contracts
[{"name": "c1", "type": "http"}]
```
"""
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "c1"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    result = extract_contracts_from_spec("No contracts here")
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = """```contracts
not valid json
```"""
    result = extract_contracts_from_spec(spec)
    assert result is None
