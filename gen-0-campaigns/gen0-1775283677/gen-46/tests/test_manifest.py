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
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64
    finally:
        tmp_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches SHA-256 of file content."""
    from src.manifest import compute_spec_hash
    content = b"my spec content"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(content)
        tmp_path = Path(f.name)
    try:
        result = compute_spec_hash(tmp_path)
        assert result == expected
    finally:
        tmp_path.unlink()


def test_artifact_hash_excludes_manifest() -> None:
    """Artifact hash excludes manifest.json."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"content")
        (root / "manifest.json").write_bytes(b'{"key": "value"}')
        files = ["src.py", "manifest.json"]
        hash_with = compute_artifact_hash(root, files)
        hash_without = compute_artifact_hash(root, ["src.py"])
        assert hash_with == hash_without


def test_artifact_hash_includes_file_content() -> None:
    """Artifact hash changes when file content changes."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"content v1")
        h1 = compute_artifact_hash(root, ["src.py"])
        (root / "src.py").write_bytes(b"content v2")
        h2 = compute_artifact_hash(root, ["src.py"])
        assert h1 != h2


def test_artifact_hash_uses_sorted_order() -> None:
    """Artifact hash uses lexicographic sort."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "b.py").write_bytes(b"b content")
        (root / "a.py").write_bytes(b"a content")
        h1 = compute_artifact_hash(root, ["a.py", "b.py"])
        h2 = compute_artifact_hash(root, ["b.py", "a.py"])
        assert h1 == h2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Without null separator, "a" + "bc" and "ab" + "c" would collide
        (root / "a").write_bytes(b"bc")
        h1 = compute_artifact_hash(root, ["a"])
        (root / "a").write_bytes(b"c")
        # Rename file to "ab"
        (root / "ab").write_bytes(b"c")
    with tempfile.TemporaryDirectory() as tmpdir2:
        root2 = Path(tmpdir2)
        (root2 / "ab").write_bytes(b"c")
        h2 = compute_artifact_hash(root2, ["ab"])
        (root2 / "a").write_bytes(b"bc")
        h3 = compute_artifact_hash(root2, ["a"])
        # These should differ because of the null separator
        assert h2 != h3


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
    """Built manifest has all required fields."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"content")
        (root / "manifest.json").write_bytes(b"{}")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="claude-test",
            token_usage={"input": 100, "output": 200},
            files=["src.py", "manifest.json"],
        )
        required = [
            "cambrian-version", "generation", "parent-generation",
            "spec-hash", "artifact-hash", "producer-model",
            "token-usage", "files", "created-at", "entry",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"


def test_build_manifest_generation_numbers() -> None:
    """Manifest has correct generation and parent."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=5,
            parent_generation=4,
            spec_hash="sha256:" + "b" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["a.py"],
        )
        assert manifest["generation"] == 5
        assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """Manifest has correct token usage."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "c" * 64,
            producer_model="model",
            token_usage={"input": 1000, "output": 2000},
            files=["a.py"],
        )
        assert manifest["token-usage"]["input"] == 1000
        assert manifest["token-usage"]["output"] == 2000


def test_build_manifest_cambrian_version() -> None:
    """Manifest has cambrian-version: 1."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "d" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["a.py"],
        )
        assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """Manifest entry points use module form for start."""
    from src.manifest import build_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "e" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["a.py"],
        )
        entry = manifest["entry"]
        assert "python -m src.prime" in entry["start"]
        assert "pytest" in entry["test"]
        assert entry["build"] != ""
        assert "8401" in entry["health"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json."""
    from src.manifest import build_manifest, write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "f" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["a.py"],
        )
        write_manifest(root, manifest)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["generation"] == 1


def test_build_manifest_with_contracts() -> None:
    """Manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [
        {"name": "health", "type": "http", "method": "GET", "path": "/health",
         "expect": {"status": 200}}
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_bytes(b"x")
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=["a.py"],
            contracts=contracts,
        )
        assert "contracts" in manifest
        assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """The artifact-hash in manifest excludes manifest.json itself."""
    from src.manifest import build_manifest, compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_bytes(b"content")
        files = ["src.py", "manifest.json"]
        manifest = build_manifest(
            artifact_root=root,
            generation=1,
            parent_generation=0,
            spec_hash="sha256:" + "a" * 64,
            producer_model="model",
            token_usage={"input": 0, "output": 0},
            files=files,
        )
        # Compute expected hash manually (without manifest.json)
        expected = compute_artifact_hash(root, ["src.py"])
        assert manifest["artifact-hash"] == expected


def test_extract_contracts_from_spec_found() -> None:
    """Extract contracts from spec with a contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = (
        "Some text\n"
        "```contracts\n"
        '[{"name": "health", "type": "http"}]\n'
        "```\n"
        "More text\n"
    )
    result = extract_contracts_from_spec(spec)
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """Returns None when no contracts block found."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts here\n"
    result = extract_contracts_from_spec(spec)
    assert result is None


def test_extract_contracts_invalid_json() -> None:
    """Returns None when contracts block has invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = (
        "```contracts\n"
        "not valid json\n"
        "```\n"
    )
    result = extract_contracts_from_spec(spec)
    assert result is None
