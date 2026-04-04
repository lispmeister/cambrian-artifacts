"""Tests for manifest building and hash computation."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
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
    """Spec hash matches manual SHA-256 computation."""
    from src.manifest import compute_spec_hash
    content = b"hello world spec"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        result = compute_spec_hash(tmp_path)
        assert result == expected
    finally:
        tmp_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_text("content")
        (root / "manifest.json").write_text('{"key": "value"}')
        files = ["src.py", "manifest.json"]
        result = compute_artifact_hash(root, files)
        # Hash should be based only on src.py
        hasher = hashlib.sha256()
        hasher.update(b"src.py")
        hasher.update(b"\0")
        hasher.update(b"content")
        expected = f"sha256:{hasher.hexdigest()}"
        assert result == expected


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash includes actual file content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("version 1")
        files = ["a.py"]
        hash1 = compute_artifact_hash(root, files)
        (root / "a.py").write_text("version 2")
        hash2 = compute_artifact_hash(root, files)
        assert hash1 != hash2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash is computed in lexicographic order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "z.py").write_text("z content")
        (root / "a.py").write_text("a content")
        files_ordered = ["a.py", "z.py"]
        files_reversed = ["z.py", "a.py"]
        hash1 = compute_artifact_hash(root, files_ordered)
        hash2 = compute_artifact_hash(root, files_reversed)
        assert hash1 == hash2  # Order of input doesn't matter, always sorted


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create files where concatenation without separator could collide
        (root / "ab.py").write_text("c")
        files = ["ab.py"]
        result = compute_artifact_hash(root, files)
        # Verify manually
        hasher = hashlib.sha256()
        hasher.update(b"ab.py")
        hasher.update(b"\0")
        hasher.update(b"c")
        expected = f"sha256:{hasher.hexdigest()}"
        assert result == expected


def test_artifact_hash_format() -> None:
    """Artifact hash has correct format."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("x")
        result = compute_artifact_hash(root, ["f.py"])
        assert result.startswith("sha256:")
        assert len(result) == 71  # "sha256:" + 64 hex chars


def test_build_manifest_has_required_fields() -> None:
    """Manifest has all required fields."""
    from src.manifest import build_manifest, compute_spec_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec_path = root / "spec.md"
        spec_path.write_text("# Spec")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=["src/__init__.py", "src/prime.py", "manifest.json", "spec/spec.md"],
            token_usage={"input": 100, "output": 200},
            spec_path=spec_path,
            spec_content="# Spec",
        )
        required = [
            "cambrian-version", "generation", "parent-generation",
            "spec-hash", "artifact-hash", "producer-model", "token-usage",
            "files", "created-at", "entry",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation numbers."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text("spec")
        manifest = build_manifest(
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=[],
            token_usage={"input": 0, "output": 0},
            spec_path=spec_path,
            spec_content="spec",
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest has correct token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text("spec")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=[],
            token_usage={"input": 1000, "output": 2000},
            spec_path=spec_path,
            spec_content="spec",
        )
        assert manifest["token-usage"]["input"] == 1000
        assert manifest["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text("spec")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=[],
            token_usage={"input": 0, "output": 0},
            spec_path=spec_path,
            spec_content="spec",
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text("spec")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=[],
            token_usage={"input": 0, "output": 0},
            spec_path=spec_path,
            spec_content="spec",
        )
        entry = manifest["entry"]
        assert "build" in entry
        assert "test" in entry
        assert "start" in entry
        assert "health" in entry
        # start MUST use module form
        assert entry["start"] == "python -m src.prime"
        assert "python src/prime.py" not in entry["start"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec_path = root / "spec.md"
        spec_path.write_text("spec")
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=["manifest.json"],
            token_usage={"input": 0, "output": 0},
            spec_path=spec_path,
            spec_content="spec",
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when spec has a contracts block."""
    from src.manifest import build_manifest
    spec_content = """# Spec

```contracts
[{"name": "health", "type": "http", "method": "GET", "path": "/health", "expect": {"status": 200}}]
```
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_path = Path(tmpdir) / "spec.md"
        spec_path.write_text(spec_content)
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_hash="sha256:" + "b" * 64,
            files=[],
            token_usage={"input": 0, "output": 0},
            spec_path=spec_path,
            spec_content=spec_content,
        )
        assert "contracts" in manifest
        assert len(manifest["contracts"]) == 1
        assert manifest["contracts"][0]["name"] == "health"


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact hash in manifest excludes manifest.json itself."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_text("code")
        (root / "manifest.json").write_text('{}')
        # Hash with manifest in list — it should be excluded
        h1 = compute_artifact_hash(root, ["src.py", "manifest.json"])
        # Hash without manifest in list
        h2 = compute_artifact_hash(root, ["src.py"])
        assert h1 == h2


def test_extract_contracts_from_spec_found() -> None:
    """Extract contracts from spec with contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = """# Spec

```contracts
[{"name": "test", "type": "http"}]
```
"""
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "test"


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    result = extract_contracts_from_spec("# No contracts here")
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block has invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = """```contracts
not valid json
```"""
    result = extract_contracts_from_spec(spec)
    assert result is None
