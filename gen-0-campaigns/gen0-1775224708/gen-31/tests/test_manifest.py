"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"spec content")
        f.flush()
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
    content = b"test spec content"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        f.flush()
        path = Path(f.name)
    try:
        h = compute_spec_hash(path)
        assert h == expected
    finally:
        path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content a")
        (root / "manifest.json").write_text('{"cambrian-version": 1}')

        files = ["a.py", "manifest.json"]
        h1 = compute_artifact_hash(root, files)

        # Hash without manifest.json in list should be the same
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 == h2


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("original")
        h1 = compute_artifact_hash(root, ["a.py"])

        (root / "a.py").write_text("changed")
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort regardless of input order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("aaa")
        (root / "b.py").write_text("bbb")

        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Without null separator, "a" + "bc" == "ab" + "c"
        # With null separator, they differ
        (root / "a").write_text("bc")
        h1 = compute_artifact_hash(root, ["a"])

        (root / "a").unlink()
        (root / "ab").write_text("c")
        h2 = compute_artifact_hash(root, ["ab"])

        assert h1 != h2


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content")
        h = compute_artifact_hash(root, ["a.py"])
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest has all MUST fields."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="claude-sonnet-4-6",
        token_usage={"input": 100, "output": 200},
        files=["src/__init__.py", "src/prime.py", "manifest.json"],
        spec_content="spec",
    )
    required_fields = [
        "cambrian-version", "generation", "parent-generation",
        "spec-hash", "artifact-hash", "producer-model", "token-usage",
        "files", "created-at", "entry",
    ]
    for field in required_fields:
        assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent-generation."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="claude-sonnet-4-6",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_content="spec",
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest token usage matches input."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="model",
        token_usage={"input": 1234, "output": 5678},
        files=["manifest.json"],
        spec_content="spec",
    )
    assert manifest["token-usage"]["input"] == 1234
    assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_content="spec",
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_content="spec",
    )
    entry = manifest["entry"]
    assert "build" in entry
    assert "test" in entry
    assert "start" in entry
    assert "health" in entry
    assert "python -m src.prime" in entry["start"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in the artifact root."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["manifest.json"],
            spec_content="spec",
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when spec has a contracts block."""
    from src.manifest import build_manifest
    spec_with_contracts = """
Some spec content.

```contracts
[
  {"name": "health", "type": "http", "method": "GET", "path": "/health",
   "expect": {"status": 200}}
]
```

More content.
"""
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_content=spec_with_contracts,
    )
    assert "contracts" in manifest
    assert len(manifest["contracts"]) == 1
    assert manifest["contracts"][0]["name"] == "health"


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """Artifact hash computation used in manifest excludes manifest.json itself."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "prime.py").write_text("print('hello')")
        (root / "manifest.json").write_text('{"cambrian-version": 1}')

        files_with = ["src/prime.py", "manifest.json"]
        files_without = ["src/prime.py"]

        h_with = compute_artifact_hash(root, files_with)
        h_without = compute_artifact_hash(root, files_without)
        assert h_with == h_without


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = """
```contracts
[{"name": "test", "type": "http"}]
```
"""
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "test"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts here."
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = """
```contracts
not valid json
```
"""
    result = extract_contracts_from_spec(spec)
    assert result is None
