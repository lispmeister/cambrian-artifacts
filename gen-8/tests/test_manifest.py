"""Tests for manifest building and hash computation."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.manifest import (
    compute_spec_hash,
    compute_artifact_hash,
    build_manifest,
    write_manifest,
    extract_contracts_from_spec,
)


def test_spec_hash_format() -> None:
    """spec-hash has sha256: prefix."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"spec content")
        path = Path(f.name)
    h = compute_spec_hash(path)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


def test_spec_hash_matches_sha256() -> None:
    """spec-hash matches actual SHA-256."""
    content = b"test spec content"
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    h = compute_spec_hash(path)
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert h == expected


def test_artifact_hash_excludes_manifest() -> None:
    """artifact-hash excludes manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"file a")
        (root / "manifest.json").write_bytes(b"manifest content")
        files = ["a.py", "manifest.json"]

        h_with = compute_artifact_hash(root, files)

        # Hash with only a.py
        hasher = hashlib.sha256()
        hasher.update(b"a.py")
        hasher.update(b"\0")
        hasher.update(b"file a")
        expected = "sha256:" + hasher.hexdigest()

        assert h_with == expected


def test_artifact_hash_includes_file_content() -> None:
    """artifact-hash includes file content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_bytes(b"hello world")
        files = ["b.py"]

        h = compute_artifact_hash(root, files)

        hasher = hashlib.sha256()
        hasher.update(b"b.py")
        hasher.update(b"\0")
        hasher.update(b"hello world")
        expected = "sha256:" + hasher.hexdigest()

        assert h == expected


def test_artifact_hash_uses_sorted_order() -> None:
    """artifact-hash uses lexicographic sort."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "z.py").write_bytes(b"z content")
        (root / "a.py").write_bytes(b"a content")
        files_unsorted = ["z.py", "a.py"]
        files_sorted = ["a.py", "z.py"]

        h_unsorted = compute_artifact_hash(root, files_unsorted)
        h_sorted = compute_artifact_hash(root, files_sorted)

        # Both should produce the same hash (sorted internally)
        assert h_unsorted == h_sorted

        # Verify it matches sorted order
        hasher = hashlib.sha256()
        hasher.update(b"a.py")
        hasher.update(b"\0")
        hasher.update(b"a content")
        hasher.update(b"z.py")
        hasher.update(b"\0")
        hasher.update(b"z content")
        expected = "sha256:" + hasher.hexdigest()
        assert h_sorted == expected


def test_artifact_hash_has_null_separator() -> None:
    """artifact-hash uses null byte separator between path and content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "file.py").write_bytes(b"content")
        files = ["file.py"]

        h = compute_artifact_hash(root, files)

        # Hash WITH null separator
        hasher = hashlib.sha256()
        hasher.update(b"file.py")
        hasher.update(b"\0")
        hasher.update(b"content")
        expected_with_null = "sha256:" + hasher.hexdigest()

        # Hash WITHOUT null separator (should differ)
        hasher2 = hashlib.sha256()
        hasher2.update(b"file.py")
        hasher2.update(b"content")
        expected_without_null = "sha256:" + hasher2.hexdigest()

        assert h == expected_with_null
        assert h != expected_without_null


def test_artifact_hash_format() -> None:
    """artifact-hash has sha256: prefix and 64 hex chars."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_bytes(b"x")
        files = ["x.py"]
        h = compute_artifact_hash(root, files)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """build_manifest produces a dict with all MUST fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"code")
        files = ["src.py", "manifest.json"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
        )

        assert "cambrian-version" in manifest
        assert "generation" in manifest
        assert "parent-generation" in manifest
        assert "spec-hash" in manifest
        assert "artifact-hash" in manifest
        assert "producer-model" in manifest
        assert "token-usage" in manifest
        assert "files" in manifest
        assert "created_at" in manifest
        assert "entry" in manifest
        assert "build" in manifest["entry"]
        assert "test" in manifest["entry"]
        assert "start" in manifest["entry"]
        assert "health" in manifest["entry"]


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent-generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        files = ["f.py"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=7,
            parent_generation=6,
            spec_hash="sha256:" + "b" * 64,
            model="model",
            input_tokens=0,
            output_tokens=0,
        )

        assert manifest["generation"] == 7
        assert manifest["parent-generation"] == 6


def test_build_manifest_token_usage() -> None:
    """Manifest token-usage has input and output fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        files = ["f.py"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            model="model",
            input_tokens=12345,
            output_tokens=6789,
        )

        assert manifest["token-usage"]["input"] == 12345
        assert manifest["token-usage"]["output"] == 6789


def test_build_manifest_cambrian_version() -> None:
    """Manifest cambrian-version is 1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        files = ["f.py"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            model="model",
            input_tokens=0,
            output_tokens=0,
        )

        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points include expected commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        files = ["f.py"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            model="model",
            input_tokens=0,
            output_tokens=0,
        )

        assert "pip install" in manifest["entry"]["build"]
        assert "pytest" in manifest["entry"]["test"]
        assert "python" in manifest["entry"]["start"]
        assert "http://localhost:8401/health" == manifest["entry"]["health"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data = {"cambrian-version": 1, "generation": 1}
        path = write_manifest(root, data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["cambrian-version"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        files = ["f.py"]
        contracts = [{"name": "health", "type": "http"}]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            model="model",
            input_tokens=0,
            output_tokens=0,
            contracts=contracts,
        )

        assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in manifest excludes manifest.json itself."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"source code")
        files = ["src.py", "manifest.json"]

        manifest = build_manifest(
            artifact_root=root,
            files=files,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            model="model",
            input_tokens=0,
            output_tokens=0,
        )

        # Recompute manually excluding manifest.json
        hasher = hashlib.sha256()
        hasher.update(b"src.py")
        hasher.update(b"\0")
        hasher.update(b"source code")
        expected = "sha256:" + hasher.hexdigest()

        assert manifest["artifact-hash"] == expected


def test_extract_contracts_from_spec_found() -> None:
    """Contracts are extracted from spec fenced block."""
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
    """Returns None when no contracts block."""
    spec = "No contracts here"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block has invalid JSON."""
    spec = (
        "```contracts\n"
        "not valid json\n"
        "```\n"
    )
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None