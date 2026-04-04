"""Tests for manifest building and hash computation."""
import hashlib
import json
import pytest
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    h = compute_spec_hash("test content")
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64  # "sha256:" + 64 hex chars


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches actual SHA-256."""
    from src.manifest import compute_spec_hash
    content = "hello world"
    expected = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    assert compute_spec_hash(content) == expected


def test_artifact_hash_excludes_manifest(tmp_path: Path) -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "a.py").write_text("content")
    (tmp_path / "manifest.json").write_text('{"key": "val"}')
    files = ["a.py", "manifest.json"]
    h1 = compute_artifact_hash(tmp_path, files)
    # Change manifest.json — hash should not change
    (tmp_path / "manifest.json").write_text('{"key": "different"}')
    h2 = compute_artifact_hash(tmp_path, files)
    assert h1 == h2


def test_artifact_hash_includes_file_content(tmp_path: Path) -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "a.py").write_text("original")
    files = ["a.py"]
    h1 = compute_artifact_hash(tmp_path, files)
    (tmp_path / "a.py").write_text("modified")
    h2 = compute_artifact_hash(tmp_path, files)
    assert h1 != h2


def test_artifact_hash_uses_sorted_order(tmp_path: Path) -> None:
    """Artifact hash uses sorted file order."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "b.py").write_text("b content")
    (tmp_path / "a.py").write_text("a content")
    files_ab = ["a.py", "b.py"]
    files_ba = ["b.py", "a.py"]
    h1 = compute_artifact_hash(tmp_path, files_ab)
    h2 = compute_artifact_hash(tmp_path, files_ba)
    assert h1 == h2  # sorted, so same order


def test_artifact_hash_has_null_separator(tmp_path: Path) -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    # Two files where path+content of one could collide with another without separator
    (tmp_path / "ab.py").write_text("c")
    (tmp_path / "a.py").write_text("bc")
    files = ["ab.py", "a.py"]
    h = compute_artifact_hash(tmp_path, files)
    # Verify by manual computation with null separator
    hasher = hashlib.sha256()
    for rel in sorted(files):
        hasher.update(rel.encode())
        hasher.update(b"\0")
        hasher.update((tmp_path / rel).read_bytes())
    expected = f"sha256:{hasher.hexdigest()}"
    assert h == expected


def test_artifact_hash_format(tmp_path: Path) -> None:
    """Artifact hash has sha256: prefix."""
    from src.manifest import compute_artifact_hash
    (tmp_path / "f.py").write_text("x")
    h = compute_artifact_hash(tmp_path, ["f.py"])
    assert h.startswith("sha256:")
    assert len(h) == 7 + 64


def test_build_manifest_has_required_fields(tmp_path: Path) -> None:
    """Built manifest has all required fields."""
    from src.manifest import build_manifest
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass")
    files = ["src/main.py", "manifest.json"]
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_root=tmp_path,
        files=files,
        model="claude-test",
        token_input=100,
        token_output=200,
    )
    assert "cambrian-version" in m
    assert "generation" in m
    assert "parent-generation" in m
    assert "spec-hash" in m
    assert "artifact-hash" in m
    assert "producer-model" in m
    assert "token-usage" in m
    assert "files" in m
    assert "created-at" in m
    assert "entry" in m


def test_build_manifest_generation_numbers(tmp_path: Path) -> None:
    """Manifest has correct generation and parent."""
    from src.manifest import build_manifest
    (tmp_path / "f.py").write_text("x")
    m = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "b" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="test",
        token_input=0,
        token_output=0,
    )
    assert m["generation"] == 5
    assert m["parent-generation"] == 4


def test_build_manifest_token_usage(tmp_path: Path) -> None:
    """Manifest token usage is correct."""
    from src.manifest import build_manifest
    (tmp_path / "f.py").write_text("x")
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "c" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="test",
        token_input=1000,
        token_output=2000,
    )
    assert m["token-usage"]["input"] == 1000
    assert m["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version(tmp_path: Path) -> None:
    """Manifest cambrian-version is 1."""
    from src.manifest import build_manifest
    (tmp_path / "f.py").write_text("x")
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "d" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="test",
        token_input=0,
        token_output=0,
    )
    assert m["cambrian-version"] == 1


def test_build_manifest_entry_points(tmp_path: Path) -> None:
    """Manifest entry points use module form."""
    from src.manifest import build_manifest
    (tmp_path / "f.py").write_text("x")
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "e" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="test",
        token_input=0,
        token_output=0,
    )
    assert m["entry"]["start"] == "python -m src.prime"
    assert "pytest" in m["entry"]["test"]
    assert m["entry"]["health"] == "http://localhost:8401/health"


def test_write_manifest_creates_file(tmp_path: Path) -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    (tmp_path / "f.py").write_text("x")
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "f" * 64,
        artifact_root=tmp_path,
        files=["f.py", "manifest.json"],
        model="test",
        token_input=0,
        token_output=0,
    )
    write_manifest(tmp_path, m)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["generation"] == 1


def test_build_manifest_with_contracts(tmp_path: Path) -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    (tmp_path / "f.py").write_text("x")
    contracts = [{"name": "health", "type": "http", "method": "GET",
                  "path": "/health", "expect": {"status": 200}}]
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "g" * 64,
        artifact_root=tmp_path,
        files=["f.py"],
        model="test",
        token_input=0,
        token_output=0,
        contracts=contracts,
    )
    assert "contracts" in m
    assert m["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest(tmp_path: Path) -> None:
    """Artifact hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, write_manifest, compute_artifact_hash
    (tmp_path / "src.py").write_text("code")
    files = ["src.py", "manifest.json"]
    m = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "h" * 64,
        artifact_root=tmp_path,
        files=files,
        model="test",
        token_input=0,
        token_output=0,
    )
    # Verify that artifact hash excludes manifest.json
    expected_hash = compute_artifact_hash(tmp_path, files)
    assert m["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """Extract contracts from spec when present."""
    from src.manifest import extract_contracts_from_spec
    spec = '''Some text
```contracts
[{"name": "health", "type": "http"}]
```
More text'''
    result = extract_contracts_from_spec(spec)
    assert result == [{"name": "health", "type": "http"}]


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns empty list when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    result = extract_contracts_from_spec("No contracts here")
    assert result == []


def test_extract_contracts_invalid_json() -> None:
    """Returns empty list when contracts JSON is invalid."""
    from src.manifest import extract_contracts_from_spec
    spec = """```contracts
not valid json
```"""
    result = extract_contracts_from_spec(spec)
    assert result == []
