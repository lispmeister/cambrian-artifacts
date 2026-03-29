"""Tests for manifest building and hash computation."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.manifest import (
    build_manifest,
    compute_artifact_hash,
    compute_spec_hash,
    extract_contracts_from_spec,
    write_manifest,
)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    """Create a temporary artifact directory with some files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / "tests" / "test_main.py").write_text("def test_it(): pass\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("aiohttp\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    """Create a temporary spec file."""
    spec = tmp_path / "spec.md"
    spec.write_text("# Spec\nThis is the spec.\n", encoding="utf-8")
    return spec


def test_spec_hash_format(spec_file: Path) -> None:
    """spec-hash has sha256: prefix."""
    h = compute_spec_hash(spec_file)
    assert h.startswith("sha256:")


def test_spec_hash_matches_sha256(spec_file: Path) -> None:
    """spec-hash matches manual SHA-256 computation."""
    content = spec_file.read_bytes()
    expected = "sha256:" + hashlib.sha256(content).hexdigest()
    assert compute_spec_hash(spec_file) == expected


def test_artifact_hash_excludes_manifest(artifact_dir: Path) -> None:
    """artifact-hash computation excludes manifest.json."""
    files_with_manifest = ["src/main.py", "requirements.txt", "manifest.json"]
    files_without_manifest = ["src/main.py", "requirements.txt"]
    (artifact_dir / "manifest.json").write_text("{}", encoding="utf-8")
    h1 = compute_artifact_hash(artifact_dir, files_with_manifest)
    h2 = compute_artifact_hash(artifact_dir, files_without_manifest)
    assert h1 == h2


def test_artifact_hash_includes_file_content(artifact_dir: Path) -> None:
    """artifact-hash changes when file content changes."""
    files = ["src/main.py", "requirements.txt"]
    h1 = compute_artifact_hash(artifact_dir, files)
    (artifact_dir / "src" / "main.py").write_text("print('changed')\n", encoding="utf-8")
    h2 = compute_artifact_hash(artifact_dir, files)
    assert h1 != h2


def test_artifact_hash_uses_sorted_order(artifact_dir: Path) -> None:
    """artifact-hash is order-independent (uses sorted order)."""
    files_ab = ["src/main.py", "requirements.txt"]
    files_ba = ["requirements.txt", "src/main.py"]
    h1 = compute_artifact_hash(artifact_dir, files_ab)
    h2 = compute_artifact_hash(artifact_dir, files_ba)
    assert h1 == h2


def test_artifact_hash_has_null_separator(artifact_dir: Path) -> None:
    """
    artifact-hash uses null byte separator between path and content.
    Verify by manually computing and comparing.
    """
    files = ["src/main.py"]
    h = compute_artifact_hash(artifact_dir, files)
    hasher = hashlib.sha256()
    hasher.update(b"src/main.py")
    hasher.update(b"\0")
    hasher.update((artifact_dir / "src/main.py").read_bytes())
    expected = f"sha256:{hasher.hexdigest()}"
    assert h == expected


def test_artifact_hash_format(artifact_dir: Path) -> None:
    """artifact-hash has sha256: prefix."""
    h = compute_artifact_hash(artifact_dir, ["src/main.py"])
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def _make_spec_in_artifact(artifact_dir: Path, spec_file: Path) -> Path:
    """Helper: copy spec file into artifact_dir/spec/ and return path."""
    spec_dest = artifact_dir / "spec"
    spec_dest.mkdir(exist_ok=True)
    shutil.copy2(spec_file, spec_dest / "spec.md")
    return spec_dest / "spec.md"


def test_build_manifest_has_required_fields(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest produces dict with all MUST fields."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)
    files = ["src/main.py", "requirements.txt", "spec/spec.md"]

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=files,
        model="claude-test",
        token_input=100,
        token_output=200,
    )

    required_fields = [
        "cambrian-version", "generation", "parent-generation", "spec-hash",
        "artifact-hash", "producer-model", "token-usage", "files",
        "created_at", "entry",
    ]
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"


def test_build_manifest_generation_numbers(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest sets generation and parent-generation correctly."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=5,
        parent_generation=4,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=100,
        token_output=200,
    )
    assert manifest["generation"] == 5
    assert manifest["parent-generation"] == 4


def test_build_manifest_token_usage(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest sets token-usage correctly."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=12345,
        token_output=67890,
    )
    assert manifest["token-usage"]["input"] == 12345
    assert manifest["token-usage"]["output"] == 67890


def test_build_manifest_cambrian_version(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest sets cambrian-version to 1."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=0,
        token_output=0,
    )
    assert manifest["cambrian-version"] == 1


def test_build_manifest_entry_points(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest includes entry points."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=0,
        token_output=0,
    )
    assert "build" in manifest["entry"]
    assert "test" in manifest["entry"]
    assert "start" in manifest["entry"]
    assert "health" in manifest["entry"]


def test_write_manifest_creates_file(artifact_dir: Path, spec_file: Path) -> None:
    """write_manifest creates manifest.json in artifact root."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=0,
        token_output=0,
    )
    write_manifest(artifact_dir, manifest)
    manifest_path = artifact_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["generation"] == 1


def test_build_manifest_with_contracts(artifact_dir: Path, spec_file: Path) -> None:
    """build_manifest includes contracts when provided."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    contracts = [{"name": "health", "type": "http", "method": "GET", "path": "/health"}]
    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=["src/main.py"],
        model="claude-test",
        token_input=0,
        token_output=0,
        contracts=contracts,
    )
    assert "contracts" in manifest
    assert manifest["contracts"] == contracts


def test_artifact_hash_in_manifest_excludes_manifest(
    artifact_dir: Path, spec_file: Path
) -> None:
    """artifact-hash in manifest excludes manifest.json."""
    spec_path = _make_spec_in_artifact(artifact_dir, spec_file)

    files = ["src/main.py", "spec/spec.md"]
    manifest = build_manifest(
        artifact_root=artifact_dir,
        generation=1,
        parent_generation=0,
        spec_path=spec_path,
        files=files,
        model="claude-test",
        token_input=0,
        token_output=0,
    )
    # Verify artifact-hash was computed without manifest.json
    expected_hash = compute_artifact_hash(artifact_dir, files)
    assert manifest["artifact-hash"] == expected_hash


def test_extract_contracts_from_spec_found() -> None:
    """extract_contracts_from_spec returns contracts array when present."""
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
    spec = "Some spec without contracts block\n"
    assert extract_contracts_from_spec(spec) is None


def test_extract_contracts_invalid_json() -> None:
    """extract_contracts_from_spec returns None for invalid JSON."""
    spec = "```contracts\nnot valid json\n```\n"
    assert extract_contracts_from_spec(spec) is None