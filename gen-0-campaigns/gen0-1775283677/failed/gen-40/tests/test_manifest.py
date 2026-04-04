"""Tests for manifest building and hash computation."""
import hashlib
import json
import tempfile
from pathlib import Path


def test_spec_hash_format() -> None:
    """Spec hash has sha256: prefix."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("spec content")
        spec_path = Path(f.name)
    try:
        h = compute_spec_hash(spec_path)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex chars
    finally:
        spec_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches expected SHA-256."""
    from src.manifest import compute_spec_hash
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
        f.write(b"test spec content")
        spec_path = Path(f.name)
    try:
        expected = "sha256:" + hashlib.sha256(b"test spec content").hexdigest()
        assert compute_spec_hash(spec_path) == expected
    finally:
        spec_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """artifact-hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content a")
        (root / "manifest.json").write_text('{"key": "value"}')
        files = ["a.py", "manifest.json"]

        h_with = compute_artifact_hash(root, files)
        h_without = compute_artifact_hash(root, ["a.py"])
        assert h_with == h_without


def test_artifact_hash_includes_file_content() -> None:
    """artifact-hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content a")
        h1 = compute_artifact_hash(root, ["a.py"])
        (root / "a.py").write_text("different content")
        h2 = compute_artifact_hash(root, ["a.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """artifact-hash uses lexicographic sort order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_text("b content")
        (root / "a.py").write_text("a content")
        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """artifact-hash uses null separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Without null separator, these two could collide:
        # file "ab" with content "c" vs file "a" with content "bc"
        (root / "ab").write_text("c")
        h1 = compute_artifact_hash(root, ["ab"])

        (root / "ab").unlink()
        (root / "a").write_text("bc")
        h2 = compute_artifact_hash(root, ["a"])

        assert h1 != h2


def test_artifact_hash_format() -> None:
    """artifact-hash has sha256: prefix and 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("content")
        h = compute_artifact_hash(root, ["a.py"])
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64


def test_build_manifest_has_required_fields() -> None:
    """build_manifest produces all MUST fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 100, "output": 200},
        )
        assert manifest["cambrian-version"] == 1
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
    """build_manifest sets generation and parent-generation correctly."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """build_manifest token-usage matches input."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 12345, "output": 67890},
        )
        assert manifest["token-usage"]["input"] == 12345
        assert manifest["token-usage"]["output"] == 67890


def test_build_manifest_cambrian_version() -> None:
    """build_manifest sets cambrian-version to 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """build_manifest entry uses module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert manifest["entry"]["start"] == "python -m src.prime"
        assert manifest["entry"]["test"] == "python -m pytest tests/ -v"
        assert manifest["entry"]["build"] == "uv pip install -r requirements.txt"


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """build_manifest includes contracts."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        assert "contracts" in manifest
        assert isinstance(manifest["contracts"], list)
        assert len(manifest["contracts"]) > 0


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """artifact-hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("code")
        manifest = build_manifest(
            artifact_root=root,
            files=["a.py"],
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 0, "output": 0},
        )
        expected_hash = compute_artifact_hash(root, ["a.py"])
        assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec finds contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = '```contracts\n[{"name": "test", "type": "http"}]\n```'
    contracts = extract_contracts_from_spec(spec)
    assert contracts is not None
    assert len(contracts) == 1
    assert contracts[0]["name"] == "test"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns None when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts here."
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = "```contracts\nnot valid json\n```"
    contracts = extract_contracts_from_spec(spec)
    assert contracts is None
