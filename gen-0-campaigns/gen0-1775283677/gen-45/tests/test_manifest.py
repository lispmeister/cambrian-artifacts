"""Tests for manifest building and hash computation."""
import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Spec")
        fname = f.name
    result = compute_spec_hash(Path(fname))
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches standard SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"# Test Spec Content"
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
        f.write(content)
        fname = f.name
    result = compute_spec_hash(Path(fname))
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert result == expected


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content a")
        (root / "manifest.json").write_bytes(b'{"generation": 1}')
        files = ["a.py", "manifest.json"]
        h1 = compute_artifact_hash(root, files)
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 == h2


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content a")
        h1 = compute_artifact_hash(root, ["a.py"])
        (root / "a.py").write_bytes(b"content b")
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_bytes(b"content b")
        (root / "a.py").write_bytes(b"content a")
        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash includes null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Two files where without null separator, hash collisions could occur
        (root / "ab").write_bytes(b"c")
        (root / "a").write_bytes(b"bc")
        h1 = compute_artifact_hash(root, ["ab"])
        h2 = compute_artifact_hash(root, ["a"])
        assert h1 != h2


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        result = compute_artifact_hash(root, ["a.py"])
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest contains all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "__init__.py").write_bytes(b"")
        files = ["src/__init__.py"]
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=files,
            producer_model="claude-test",
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
    """Manifest generation numbers are correct."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 10, "output": 20},
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
            spec_hash="sha256:" + "c" * 64,
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
            spec_hash="sha256:" + "d" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use correct forms."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"content")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["entry"]["start"] == "python -m src.prime"
        assert "pytest" in manifest["entry"]["test"]
        assert manifest["entry"]["health"] == "http://localhost:8401/health"


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
            spec_hash="sha256:" + "f" * 64,
            files=["a.py"],
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        assert (root / "manifest.json").exists()
        data = json.loads((root / "manifest.json").read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts from spec when present."""
    from src.manifest import build_manifest
    spec_with_contracts = """
Some spec content.

```contracts
[{"name": "health-check", "type": "http", "method": "GET", "path": "/health"}]
```

More content.
"""
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
            spec_content=spec_with_contracts,
        )
        assert "contracts" in manifest
        assert manifest["contracts"][0]["name"] == "health-check"


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in the manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, write_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"some content")
        files = ["a.py"]
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            files=files,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        # Now compute hash with manifest.json included in file list
        all_files = ["a.py", "manifest.json"]
        hash_with_manifest = compute_artifact_hash(root, all_files)
        hash_without_manifest = compute_artifact_hash(root, ["a.py"])
        # Both should equal the manifest's artifact-hash (since manifest.json is excluded)
        assert manifest["artifact-hash"] == hash_without_manifest
        assert manifest["artifact-hash"] == hash_with_manifest


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = '```contracts\n[{"name": "test"}]\n```'
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert result[0]["name"] == "test"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    result = extract_contracts_from_spec("# Just a spec\nNo contracts here.")
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None
