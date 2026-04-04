"""Tests for manifest building and hash computation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash starts with 'sha256:'."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
        f.write(b"test spec content")
        tmp = Path(f.name)
    try:
        h = compute_spec_hash(tmp)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # 'sha256:' + 64 hex chars
    finally:
        tmp.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"spec content for hashing"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
        f.write(content)
        tmp = Path(f.name)
    try:
        h = compute_spec_hash(tmp)
        assert h == expected
    finally:
        tmp.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"source")
        (root / "manifest.json").write_bytes(b'{"key": "value"}')
        files = ["src.py", "manifest.json"]
        h1 = compute_artifact_hash(root, files)

        # Change manifest — hash should not change
        (root / "manifest.json").write_bytes(b'{"key": "different"}')
        h2 = compute_artifact_hash(root, files)
        assert h1 == h2


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"content1")
        h1 = compute_artifact_hash(root, ["src.py"])

        (root / "src.py").write_bytes(b"content2")
        h2 = compute_artifact_hash(root, ["src.py"])

        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses sorted file order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_bytes(b"b")
        (root / "a.py").write_bytes(b"a")

        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2  # Order of input doesn't matter, sorted internally


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Without null separator: "a" + "bc" == "ab" + "c" would collide
        (root / "a").write_bytes(b"bc")
        h1 = compute_artifact_hash(root, ["a"])

        # This verifies that null separator is used
        hasher = hashlib.sha256()
        hasher.update(b"a")
        hasher.update(b"\0")
        hasher.update(b"bc")
        expected = f"sha256:{hasher.hexdigest()}"
        assert h1 == expected


def test_artifact_hash_format() -> None:
    """Artifact hash starts with 'sha256:' and has 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "file.py").write_bytes(b"content")
        h = compute_artifact_hash(root, ["file.py"])
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """build_manifest returns dict with all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"source")
        manifest = build_manifest(
            artifact_root=root,
            files=["src.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            spec_content="spec",
            token_usage={"input": 100, "output": 200},
            model="claude-test",
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
    """build_manifest uses correct generation and parent-generation."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """build_manifest includes token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            spec_content="spec",
            token_usage={"input": 1000, "output": 2000},
            model="m",
        )
        assert manifest["token-usage"]["input"] == 1000
        assert manifest["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version() -> None:
    """build_manifest sets cambrian-version to 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """build_manifest includes required entry points."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        assert "build" in manifest["entry"]
        assert "test" in manifest["entry"]
        assert "start" in manifest["entry"]
        assert "health" in manifest["entry"]
        # entry.start MUST use module form
        assert manifest["entry"]["start"] == "python -m src.prime"


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in artifact root."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts array."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            files=["f.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        assert "contracts" in manifest
        assert isinstance(manifest["contracts"], list)


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"source code")
        files = ["src.py", "manifest.json"]
        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "0" * 64,
            spec_content="spec",
            token_usage={"input": 0, "output": 0},
            model="m",
        )
        expected_hash = compute_artifact_hash(root, files)
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts JSON block."""
    from src.manifest import extract_contracts_from_spec
    spec = (
        "Some text\n"
        "```contracts\n"
        '[{"name": "health", "type": "http"}]\n'
        "```\n"
        "More text\n"
    )
    contracts = extract_contracts_from_spec(spec)
    assert contracts is not None
    assert len(contracts) == 1
    assert contracts[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts block here."
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None
