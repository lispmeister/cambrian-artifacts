"""Tests for manifest building and hash computation."""
import hashlib
import json
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
        assert len(result) == len("sha256:") + 64
    finally:
        tmp_path.unlink()


def test_spec_hash_matches_sha256() -> None:
    """Spec hash matches manual SHA-256 computation."""
    from src.manifest import compute_spec_hash
    content = b"my spec content"
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".md", delete=False) as f:
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
        (root / "src.py").write_text("code")
        (root / "manifest.json").write_text('{"generation": 1}')

        files_with = ["src.py", "manifest.json"]
        files_without = ["src.py"]

        hash_with = compute_artifact_hash(root, files_with)
        hash_without = compute_artifact_hash(root, files_without)

        assert hash_with == hash_without


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
    """Artifact hash is the same regardless of file list order."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "a.py").write_text("aaa")
        (root / "b.py").write_text("bbb")

        hash1 = compute_artifact_hash(root, ["a.py", "b.py"])
        hash2 = compute_artifact_hash(root, ["b.py", "a.py"])

        assert hash1 == hash2


def test_artifact_hash_has_null_separator() -> None:
    """Artifact hash uses null byte separator between path and content."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Two files: "ab" + "c" vs "a" + "bc" — without null separator these collide
        (root / "a").write_text("bc")
        (root / "ab").write_text("c")

        hash_a = compute_artifact_hash(root, ["a"])
        hash_ab = compute_artifact_hash(root, ["ab"])

        # They should be different (null separator prevents collision)
        assert hash_a != hash_ab


def test_artifact_hash_format() -> None:
    """Artifact hash has sha256: prefix and 64 hex chars."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "f.py").write_text("content")
        result = compute_artifact_hash(root, ["f.py"])
        assert result.startswith("sha256:")
        assert len(result) == len("sha256:") + 64


def test_build_manifest_has_required_fields() -> None:
    """build_manifest returns dict with all MUST fields."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="claude-sonnet-4-6",
        token_usage={"input": 1000, "output": 500},
        files=["src/__init__.py", "src/prime.py", "manifest.json"],
        spec_file_path="spec/CAMBRIAN-SPEC-005.md",
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
    assert "build" in manifest["entry"]
    assert "test" in manifest["entry"]
    assert "start" in manifest["entry"]
    assert "health" in manifest["entry"]


def test_build_manifest_generation_numbers() -> None:
    """build_manifest sets generation and parent-generation correctly."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=5,
        parent_generation=4,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test-model",
        token_usage={"input": 100, "output": 50},
        files=["manifest.json"],
        spec_file_path="spec/spec.md",
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage() -> None:
    """build_manifest includes token usage."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test-model",
        token_usage={"input": 12345, "output": 6789},
        files=["manifest.json"],
        spec_file_path="spec/spec.md",
    )
    assert manifest["token-usage"]["input"] == 12345
    assert manifest["token-usage"]["output"] == 6789


def test_build_manifest_cambrian_version() -> None:
    """build_manifest sets cambrian-version to 1."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test-model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_file_path="spec/spec.md",
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points() -> None:
    """build_manifest uses module form for entry.start."""
    from src.manifest import build_manifest
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test-model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_file_path="spec/spec.md",
    )
    assert "python -m src.prime" in manifest["entry"]["start"]
    assert "pytest" in manifest["entry"]["test"]


def test_write_manifest_creates_file() -> None:
    """write_manifest creates manifest.json in the artifact root."""
    from src.manifest import write_manifest
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data = {"cambrian-version": 1, "generation": 1}
        write_manifest(root, data)
        manifest_path = root / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["cambrian-version"] == 1


def test_build_manifest_with_contracts() -> None:
    """build_manifest includes contracts when provided."""
    from src.manifest import build_manifest
    contracts = [{"name": "health", "type": "http", "method": "GET", "path": "/health",
                  "expect": {"status": 200}}]
    manifest = build_manifest(
        generation=1,
        parent_generation=0,
        spec_hash="sha256:" + "a" * 64,
        artifact_hash="sha256:" + "b" * 64,
        producer_model="test-model",
        token_usage={"input": 0, "output": 0},
        files=["manifest.json"],
        spec_file_path="spec/spec.md",
        contracts=contracts,
    )
    assert "contracts" in manifest
    assert manifest["contracts"][0]["name"] == "health"


def test_artifact_hash_in_manifest_excludes_manifest() -> None:
    """Artifact hash computed for manifest excludes manifest.json itself."""
    from src.manifest import compute_artifact_hash
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src.py").write_text("code here")
        (root / "manifest.json").write_text("{}")

        hash1 = compute_artifact_hash(root, ["src.py", "manifest.json"])
        hash2 = compute_artifact_hash(root, ["src.py"])
        assert hash1 == hash2


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec returns contracts from spec."""
    from src.manifest import extract_contracts_from_spec
    spec = """
Some text.

```contracts
[{"name": "health", "type": "http"}]
```

More text.
"""
    contracts = extract_contracts_from_spec(spec)
    assert len(contracts) == 1
    assert contracts[0]["name"] == "health"


def test_extract_contracts_from_spec_not_found() -> None:
    """extract_contracts_from_spec returns empty list when no contracts block."""
    from src.manifest import extract_contracts_from_spec
    spec = "No contracts here."
    contracts = extract_contracts_from_spec(spec)
    assert contracts == []


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns empty list on invalid JSON."""
    from src.manifest import extract_contracts_from_spec
    spec = """
```contracts
not valid json {{{
```
"""
    contracts = extract_contracts_from_spec(spec)
    assert contracts == []
