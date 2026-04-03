"""Tests for manifest building and hash computation."""
from __future__ import annotations

import json
from pathlib import Path


def test_spec_hash_format(tmp_path: Path) -> None:
    """spec-hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    spec = tmp_path / "spec.md"
    spec.write_text("test spec content")
    h = compute_spec_hash(spec)
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_spec_hash_matches_sha256(tmp_path: Path) -> None:
    """spec-hash matches actual SHA-256."""
    from src.manifest import compute_spec_hash
    import hashlib
    content = "test spec content"
    spec = tmp_path / "spec.md"
    spec.write_text(content)
    expected = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    assert compute_spec_hash(spec) == expected


def test_artifact_hash_excludes_manifest(tmp_path: Path) -> None:
    """artifact-hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "a.py").write_text("content a")
    (tmp_path / "manifest.json").write_text("{}")
    files = ["a.py", "manifest.json"]
    h1 = compute_artifact_hash(tmp_path, files)
    h2 = compute_artifact_hash(tmp_path, ["a.py"])
    assert h1 == h2


def test_artifact_hash_includes_file_content(tmp_path: Path) -> None:
    """Different file content produces different hash."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "a.py").write_text("content1")
    h1 = compute_artifact_hash(tmp_path, ["a.py"])
    (tmp_path / "a.py").write_text("content2")
    h2 = compute_artifact_hash(tmp_path, ["a.py"])
    assert h1 != h2


def test_artifact_hash_uses_sorted_order(tmp_path: Path) -> None:
    """Files are sorted lexicographically for hashing."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")
    h1 = compute_artifact_hash(tmp_path, ["a.py", "b.py"])
    h2 = compute_artifact_hash(tmp_path, ["b.py", "a.py"])
    assert h1 == h2


def test_artifact_hash_has_null_separator(tmp_path: Path) -> None:
    """Artifact hash uses null byte separator."""
    from src.manifest import compute_artifact_hash
    import hashlib
    (tmp_path / "a.py").write_text("content")
    h = compute_artifact_hash(tmp_path, ["a.py"])
    hasher = hashlib.sha256()
    hasher.update(b"a.py")
    hasher.update(b"\0")
    hasher.update(b"content")
    expected = f"sha256:{hasher.hexdigest()}"
    assert h == expected


def test_artifact_hash_format(tmp_path: Path) -> None:
    """artifact-hash has sha256: prefix."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "a.py").write_text("x")
    h = compute_artifact_hash(tmp_path, ["a.py"])
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """Manifest has all required fields."""
    from src.manifest import build_manifest
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:abc123" + "0" * 58,
        artifact_hash="sha256:def456" + "0" * 58,
        producer_model="claude-sonnet-4-6",
        token_usage={"input": 100, "output": 200},
        files=["src/__init__.py", "src/prime.py", "manifest.json"],
    )
    required = [
        "cambrian-version", "generation", "parent-generation",
        "spec-hash", "artifact-hash", "producer-model",
        "token-usage", "files", "created-at", "entry",
    ]
    for key in required:
        assert key in m, f"Missing required field: {key}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation numbers."""
    from src.manifest import build_manifest
    m = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    assert m["generation"] == 5
    assert m["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest has correct token usage."""
    from src.manifest import build_manifest
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 1000, "output": 2000},
        files=[],
    )
    assert m["token-usage"]["input"] == 1000
    assert m["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version() -> None:
    """Manifest has cambrian-version 1."""
    from src.manifest import build_manifest
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    assert m["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points are correct."""
    from src.manifest import build_manifest
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
    )
    entry = m["entry"]
    assert entry["build"] == "uv pip install -r requirements.txt"
    assert entry["test"] == "python -m pytest tests/ -v"
    assert entry["start"] == "python -m src.prime"
    assert entry["health"] == "http://localhost:8401/health"


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import write_manifest
    manifest = {"cambrian-version": 1, "generation": 1}
    write_manifest(tmp_path, manifest)
    assert (tmp_path / "manifest.json").exists()
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [{"name": "test", "type": "http"}]
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test",
        token_usage={"input": 0, "output": 0},
        files=[],
        contracts=contracts,
    )
    assert m["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest(tmp_path: Path) -> None:
    """Verify that artifact hash computation excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("code")
    (tmp_path / "manifest.json").write_text('{"test": true}')
    files_with = ["src/main.py", "manifest.json"]
    files_without = ["src/main.py"]
    assert compute_artifact_hash(tmp_path, files_with) == compute_artifact_hash(tmp_path, files_without)


def test_extract_contracts_from_spec_found() -> None:
    """Extract contracts from spec with contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = '```contracts\n[{"name": "test"}]\n```'
    result = extract_contracts_from_spec(spec)
    assert result == [{"name": "test"}]


def test_extract_contracts_from_spec_not_found() -> None:
    """No contracts block returns None."""
    from src.manifest import extract_contracts_from_spec
    assert extract_contracts_from_spec("no contracts here") is None


def test_extract_contracts_invalid_json() -> None:
    """Invalid JSON in contracts block returns None."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot json\n```"
    assert extract_contracts_from_spec(spec) is None
