"""Tests for manifest building and hash computation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from src.manifest import (
    compute_spec_hash,
    compute_artifact_hash,
    build_manifest,
    write_manifest,
    extract_contracts_from_spec,
)


# ---------------------------------------------------------------------------
# Hash computation tests
# ---------------------------------------------------------------------------


def test_spec_hash_format() -> None:
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
        f.write(b"# Spec content")
        tmp_path = Path(f.name)
    try:
        h = compute_spec_hash(tmp_path)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64
    finally:
        tmp_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    content = b"Hello, spec!"
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        expected = "sha256:" + hashlib.sha256(content).hexdigest()
        assert compute_spec_hash(tmp_path) == expected
    finally:
        tmp_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("print('a')")
        (root / "manifest.json").write_text('{"cambrian-version": 1}')

        files = ["a.py", "manifest.json"]
        h1 = compute_artifact_hash(root, files)

        # Changing manifest.json should not change the hash
        (root / "manifest.json").write_text('{"cambrian-version": 2}')
        h2 = compute_artifact_hash(root, files)
        assert h1 == h2


def test_artifact_hash_includes_file_content() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("version 1")
        files = ["a.py"]
        h1 = compute_artifact_hash(root, files)

        (root / "a.py").write_text("version 2")
        h2 = compute_artifact_hash(root, files)
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "z.py").write_text("z")
        (root / "a.py").write_text("a")
        files_order1 = ["z.py", "a.py"]
        files_order2 = ["a.py", "z.py"]
        h1 = compute_artifact_hash(root, files_order1)
        h2 = compute_artifact_hash(root, files_order2)
        assert h1 == h2  # sorted, so same


def test_artifact_hash_has_null_separator() -> None:
    """
    Test that the null byte separator is critical.
    Without it, ("ab", "c") and ("a", "bc") would hash the same.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create two scenarios that would collide without null separator
        (root / "ab").write_text("c")
        h1 = compute_artifact_hash(root, ["ab"])

        (root / "ab").unlink()
        (root / "a").write_text("bc")
        h2 = compute_artifact_hash(root, ["a"])

        # With null separator these should NOT be equal
        assert h1 != h2


def test_artifact_hash_format() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "file.py").write_text("content")
        h = compute_artifact_hash(root, ["file.py"])
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# build_manifest tests
# ---------------------------------------------------------------------------


def test_build_manifest_has_required_fields() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "file.py").write_text("pass")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["file.py", "manifest.json"],
            producer_model="claude-test",
            token_input=100,
            token_output=200,
        )

    required_fields = [
        "cambrian-version",
        "generation",
        "parent-generation",
        "spec-hash",
        "artifact-hash",
        "producer-model",
        "token-usage",
        "files",
        "created_at",
        "entry",
    ]
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"


def test_build_manifest_generation_numbers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_text("x")

        manifest = build_manifest(
            generation=3,
            parent_generation=2,
            spec_hash="sha256:" + "b" * 64,
            artifact_root=root,
            files=["x.py"],
            producer_model="model",
            token_input=50,
            token_output=100,
        )

    assert manifest["generation"] == 3
    assert manifest["parent-generation"] == 2


def test_build_manifest_token_usage() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_text("x")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            artifact_root=root,
            files=["x.py"],
            producer_model="model",
            token_input=1234,
            token_output=5678,
        )

    assert manifest["token-usage"]["input"] == 1234
    assert manifest["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_text("x")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            artifact_root=root,
            files=["x.py"],
            producer_model="model",
            token_input=0,
            token_output=0,
        )

    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_text("x")

        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            artifact_root=root,
            files=["x.py"],
            producer_model="model",
            token_input=0,
            token_output=0,
        )

    assert "build" in manifest["entry"]
    assert "test" in manifest["entry"]
    assert "start" in manifest["entry"]
    assert "health" in manifest["entry"]


def test_write_manifest_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest = {"cambrian-version": 1, "generation": 1}
        write_manifest(root, manifest)
        assert (root / "manifest.json").exists()
        loaded = json.loads((root / "manifest.json").read_text())
        assert loaded["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "x.py").write_text("x")

        contracts = [
            {
                "name": "health",
                "type": "http",
                "method": "GET",
                "path": "/health",
                "expect": {"status": 200, "body": {"ok": True}},
            }
        ]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            artifact_root=root,
            files=["x.py"],
            producer_model="model",
            token_input=0,
            token_output=0,
            contracts=contracts,
        )

    assert "contracts" in manifest
    assert len(manifest["contracts"]) == 1


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in manifest should not include manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_text("code here")

        files = ["src.py", "manifest.json"]
        manifest = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=files,
            producer_model="model",
            token_input=0,
            token_output=0,
        )

        artifact_hash = manifest["artifact-hash"]

        # Verify it matches direct computation (which excludes manifest.json)
        expected = compute_artifact_hash(root, files)
        assert artifact_hash == expected


# ---------------------------------------------------------------------------
# Contract extraction tests
# ---------------------------------------------------------------------------


def test_extract_contracts_from_spec_found() -> None:
    spec = """
Some spec content.

```contracts
[{"name": "health", "type": "http", "method": "GET", "path": "/health", "expect": {"status": 200}}]
```

More content.
"""
    contracts = extract_contracts_from_spec(spec)
    assert contracts is not None
    assert len(contracts) == 1
    assert contracts[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    spec = "# Spec\n\nNo contracts here.\n"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None


def test_extract_contracts_invalid_json() -> None:
    spec = "```contracts\nnot valid json\n```\n"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None