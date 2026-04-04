"""Tests for manifest building and hash computation."""

import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"test spec content")
        path = Path(f.name)
    try:
        h = compute_spec_hash(path)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64
    finally:
        path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash value matches hashlib SHA-256."""
    from src.manifest import compute_spec_hash
    content = b"my spec content here"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        assert compute_spec_hash(path) == expected
    finally:
        path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """artifact-hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"code")
        (root / "manifest.json").write_bytes(b"manifest content")
        files_with = ["src.py", "manifest.json"]
        files_without = ["src.py"]
        hash_with = compute_artifact_hash(root, files_with)
        hash_without = compute_artifact_hash(root, files_without)
        assert hash_with == hash_without


def test_artifact_hash_includes_file_content() -> None:
    """artifact-hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"version 1")
        h1 = compute_artifact_hash(root, ["a.py"])
        (root / "a.py").write_bytes(b"version 2")
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """artifact-hash is computed in lexicographic order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_bytes(b"b content")
        (root / "a.py").write_bytes(b"a content")
        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """artifact-hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create files where path+content could collide without separator
        (root / "ab").write_bytes(b"c")
        (root / "a").write_bytes(b"bc")
        # They should have different hashes due to null separator
        h1 = compute_artifact_hash(root, ["ab"])
        h2 = compute_artifact_hash(root, ["a"])
        assert h1 != h2


def test_artifact_hash_format() -> None:
    """artifact-hash has sha256: prefix and 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"content")
        h = compute_artifact_hash(root, ["f.py"])
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64


def test_build_manifest_has_required_fields() -> None:
    """build_manifest produces all MUST fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"code")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["src.py"],
            producer_model="claude-test",
            token_usage={"input": 100, "output": 200},
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
        assert "build" in m["entry"]
        assert "test" in m["entry"]
        assert "start" in m["entry"]
        assert "health" in m["entry"]


def test_build_manifest_generation_numbers() -> None:
    """build_manifest records correct generation numbers."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
        )
        assert m["generation"] == 5
        assert m["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """build_manifest records token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 1234, "output": 5678},
        )
        assert m["token-usage"]["input"] == 1234
        assert m["token-usage"]["output"] == 5678


def test_build_manifest_cambrian_version() -> None:
    """build_manifest sets cambrian-version to 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
        )
        assert m["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """build_manifest entry.start uses module form."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
        )
        assert m["entry"]["start"] == "python -m src.prime"
        assert "python -m pytest" in m["entry"]["test"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, m)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """build_manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [{"name": "health", "type": "http", "method": "GET", "path": "/health"}]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_bytes(b"x")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["f.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            contracts=contracts,
        )
        assert "contracts" in m
        assert m["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """artifact-hash in manifest excludes manifest.json."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "code.py").write_bytes(b"source code")
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            artifact_root=root,
            files=["code.py"],
            producer_model="model",
            token_usage={"input": 0, "output": 0},
        )
        # The hash in the manifest should match hash computed without manifest.json
        expected_hash = compute_artifact_hash(root, ["code.py"])
        assert m["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = '```contracts\n[{"name": "health", "type": "http"}]\n```'
    contracts = extract_contracts_from_spec(spec)
    assert contracts is not None
    assert len(contracts) == 1
    assert contracts[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "# Spec\n\nNo contracts here.\n"
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None on invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    result = extract_contracts_from_spec(spec)
    assert result is None
