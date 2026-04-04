"""Tests for manifest building and hash computation."""
import hashlib
import json
import tempfile
from pathlib import Path
import pytest


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("spec content")
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        tmp_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"hello world spec"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        assert result == expected
    finally:
        tmp_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash does not include manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content a")
        (root / "manifest.json").write_text('{"key": "value"}')
        files = ["a.py", "manifest.json"]

        hash_with = compute_artifact_hash(root, files)

        # Hash without manifest should be the same
        files_no_manifest = ["a.py"]
        hash_without = compute_artifact_hash(root, files_no_manifest)

        assert hash_with == hash_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("original content")
        files = ["a.py"]
        hash1 = compute_artifact_hash(root, files)

        (root / "a.py").write_text("changed content")
        hash2 = compute_artifact_hash(root, files)

        assert hash1 != hash2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "z.py").write_text("z content")
        (root / "a.py").write_text("a content")
        files_forward = ["a.py", "z.py"]
        files_reverse = ["z.py", "a.py"]

        hash1 = compute_artifact_hash(root, files_forward)
        hash2 = compute_artifact_hash(root, files_reverse)

        assert hash1 == hash2  # Same result regardless of input order


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Verify the null separator is used by checking against manual computation
        (root / "f.txt").write_text("data")
        files = ["f.txt"]
        result = compute_artifact_hash(root, files)

        # Manual computation with null separator
        hasher = hashlib.sha256()
        hasher.update(b"f.txt")
        hasher.update(b"\0")
        hasher.update(b"data")
        expected = f"sha256:{hasher.hexdigest()}"
        assert result == expected


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("content")
        result = compute_artifact_hash(root, ["f.py"])
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Build manifest includes all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "__init__.py").write_text("")
        files = ["src/__init__.py"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=files,
            model="claude-sonnet-4-6",
            token_usage={"input": 1000, "output": 500},
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
    """Manifest has correct generation numbers."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 100, "output": 50},
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest has correct token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 12345, "output": 6789},
        )
        assert manifest["token-usage"]["input"] == 12345
        assert manifest["token-usage"]["output"] == 6789


def test_build_manifest_cambrian_version() -> None:
    """Manifest has cambrian-version: 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["entry"]["start"] == "python -m src.prime"
        assert manifest["entry"]["test"] == "python -m pytest tests/ -v"
        assert manifest["entry"]["build"] == "uv pip install -r requirements.txt"
        assert "health" in manifest["entry"]


def test_write_manifest_creates_file() -> None:
    """Write manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        assert (root / "manifest.json").exists()
        data = json.loads((root / "manifest.json").read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [
        {"name": "health-liveness", "type": "http", "method": "GET", "path": "/health",
         "expect": {"status": 200}}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["f.py"],
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
            contracts=contracts,
        )
        assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """artifact-hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_text("code")
        files = ["src.py"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=files,
            model="claude-sonnet-4-6",
            token_usage={"input": 0, "output": 0},
        )
        expected_hash = compute_artifact_hash(root, files)
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """Extract contracts from spec when present."""
    from src.manifest import extract_contracts_from_spec
    spec = """Some text

```contracts
[{"name": "health", "type": "http"}]
```

More text"""
    result = extract_contracts_from_spec(spec)
    assert result == [{"name": "health", "type": "http"}]


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns None when no contracts block in spec."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts here."
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block has invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = """```contracts
not valid json
```"""
    result = extract_contracts_from_spec(spec)
    assert result is None
