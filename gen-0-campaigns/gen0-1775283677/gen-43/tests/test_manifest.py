"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("spec content")
        path = Path(f.name)
    try:
        result = compute_spec_hash(path)
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches actual SHA-256 of the file."""
    from src.manifest import compute_spec_hash
    content = b"hello world spec"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        result = compute_spec_hash(path)
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

        # Hash with manifest excluded
        result = compute_artifact_hash(root, files)

        # Manual hash of only a.py
        hasher = hashlib.sha256()
        hasher.update("a.py".encode())
        hasher.update(b"\0")
        hasher.update(b"content a")
        expected = f"sha256:{hasher.hexdigest()}"
        assert result == expected


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("version 1")
        hash1 = compute_artifact_hash(root, ["a.py"])
        (root / "a.py").write_text("version 2")
        hash2 = compute_artifact_hash(root, ["a.py"])
        assert hash1 != hash2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort of file paths."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_text("b content")
        (root / "a.py").write_text("a content")

        hash1 = compute_artifact_hash(root, ["a.py", "b.py"])
        hash2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert hash1 == hash2  # Order of input doesn't matter — sorted internally


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create two files where path+content could collide without separator
        (root / "a.py").write_text("b")
        hash_with_sep = compute_artifact_hash(root, ["a.py"])

        # Manually compute without separator to verify they differ
        hasher_no_sep = hashlib.sha256()
        hasher_no_sep.update(b"a.py")
        hasher_no_sep.update(b"b")
        hash_no_sep = f"sha256:{hasher_no_sep.hexdigest()}"

        assert hash_with_sep != hash_no_sep


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix and correct length."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content")
        result = compute_artifact_hash(root, ["a.py"])
        assert result.startswith("sha256:")
        assert len(result) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Built manifest has all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec content")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc123" + "0" * 58,
            producer_model="claude-test",
            token_usage={"input": 100, "output": 200},
            spec_path=spec_file,
        )

        required = [
            "cambrian-version", "generation", "parent-generation",
            "spec-hash", "artifact-hash", "producer-model", "token-usage",
            "files", "created-at", "entry",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent-generation."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )

        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest has correct token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 1000, "output": 2000},
            spec_path=spec_file,
        )

        assert manifest["token-usage"]["input"] == 1000
        assert manifest["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version() -> None:
    """Manifest has cambrian-version: 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )

        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )

        assert manifest["entry"]["start"] == "python -m src.prime"
        assert manifest["entry"]["build"] == "uv pip install -r requirements.txt"
        assert manifest["entry"]["test"] == "python -m pytest tests/ -v"


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in artifact root."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )
        write_manifest(root, manifest)

        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec without contracts block")

        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )

        assert "contracts" in manifest
        assert isinstance(manifest["contracts"], list)


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """artifact-hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        spec_file = root / "spec.md"
        spec_file.write_text("spec")

        files = ["a.py"]
        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
            spec_path=spec_file,
        )

        expected_hash = compute_artifact_hash(root, files)
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    contracts_json = '[{"name": "test", "type": "http"}]'
    spec_content = f"# Spec\n\n```contracts\n{contracts_json}\n```\n"
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(spec_content)
        path = Path(f.name)
    try:
        result = extract_contracts_from_spec(path)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "test"
    finally:
        path.unlink()


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("# Spec\n\nNo contracts here.\n")
        path = Path(f.name)
    try:
        result = extract_contracts_from_spec(path)
        assert result is None
    finally:
        path.unlink()


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write("```contracts\nnot valid json\n```\n")
        path = Path(f.name)
    try:
        result = extract_contracts_from_spec(path)
        assert result is None
    finally:
        path.unlink()
