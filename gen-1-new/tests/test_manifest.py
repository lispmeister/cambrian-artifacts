"""Tests for manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from src.manifest import compute_spec_hash, compute_artifact_hash, build_manifest, write_manifest


@pytest.fixture
def tmp_artifact(tmp_path: Path) -> Path:
    """Create a minimal artifact directory with a few source files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "requirements.txt").write_text("aiohttp>=3.9\n")
    return tmp_path


class TestComputeSpecHash:
    def test_matches_sha256_of_file_bytes(self, tmp_path: Path):
        spec = tmp_path / "spec.md"
        spec.write_text("# My Spec")
        expected = "sha256:" + hashlib.sha256(b"# My Spec").hexdigest()
        assert compute_spec_hash(spec) == expected

    def test_has_sha256_prefix(self, tmp_path: Path):
        spec = tmp_path / "spec.md"
        spec.write_text("content")
        result = compute_spec_hash(spec)
        assert result.startswith("sha256:")

    def test_64_hex_chars_after_prefix(self, tmp_path: Path):
        spec = tmp_path / "spec.md"
        spec.write_text("content")
        result = compute_spec_hash(spec)
        hex_part = result[len("sha256:"):]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)


class TestComputeArtifactHash:
    def test_excludes_manifest_json(self, tmp_artifact: Path):
        """manifest.json must be excluded from artifact hash."""
        (tmp_artifact / "manifest.json").write_text("{}")
        files_with = ["src/main.py", "requirements.txt", "manifest.json"]
        files_without = ["src/main.py", "requirements.txt"]
        h1 = compute_artifact_hash(tmp_artifact, files_with)
        h2 = compute_artifact_hash(tmp_artifact, files_without)
        assert h1 == h2

    def test_deterministic(self, tmp_artifact: Path):
        files = ["src/main.py", "requirements.txt"]
        h1 = compute_artifact_hash(tmp_artifact, files)
        h2 = compute_artifact_hash(tmp_artifact, files)
        assert h1 == h2

    def test_order_independent(self, tmp_artifact: Path):
        """Hash should be the same regardless of file list order (sort is applied)."""
        files_ab = ["src/main.py", "requirements.txt"]
        files_ba = ["requirements.txt", "src/main.py"]
        h1 = compute_artifact_hash(tmp_artifact, files_ab)
        h2 = compute_artifact_hash(tmp_artifact, files_ba)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_artifact: Path):
        files = ["src/main.py", "requirements.txt"]
        h1 = compute_artifact_hash(tmp_artifact, files)
        (tmp_artifact / "src" / "main.py").write_text("print('different')")
        h2 = compute_artifact_hash(tmp_artifact, files)
        assert h1 != h2

    def test_includes_file_paths_in_hash(self, tmp_artifact: Path):
        """Renaming a file changes the hash even if content is identical."""
        (tmp_artifact / "src" / "other.py").write_text("print('hello')")
        h1 = compute_artifact_hash(tmp_artifact, ["src/main.py"])
        h2 = compute_artifact_hash(tmp_artifact, ["src/other.py"])
        assert h1 != h2, "Hash must differ when file path differs, even with same content"

    def test_null_separator_prevents_collision(self, tmp_artifact: Path):
        """The \\0 separator between path and content must prevent (path='ab', content='c')
        from colliding with (path='a', content='bc')."""
        # File "ab.py" with content "c" vs file "a.py" with content "bc"
        (tmp_artifact / "ab.py").write_bytes(b"c")
        (tmp_artifact / "a.py").write_bytes(b"bc")
        h1 = compute_artifact_hash(tmp_artifact, ["ab.py"])
        h2 = compute_artifact_hash(tmp_artifact, ["a.py"])
        assert h1 != h2, "Null separator must prevent path/content boundary collisions"

    def test_has_sha256_prefix(self, tmp_artifact: Path):
        result = compute_artifact_hash(tmp_artifact, ["src/main.py"])
        assert result.startswith("sha256:")


class TestBuildManifest:
    def test_has_all_required_fields(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc123",
            artifact_dir=tmp_artifact,
            files=["src/main.py", "requirements.txt"],
            producer_model="claude-opus-4-6",
            input_tokens=1000,
            output_tokens=500,
        )
        assert m["cambrian-version"] == 1
        assert m["generation"] == 1
        assert m["parent-generation"] == 0
        assert m["spec-hash"] == "sha256:abc123"
        assert m["artifact-hash"].startswith("sha256:")
        assert m["producer-model"] == "claude-opus-4-6"
        assert m["token-usage"]["input"] == 1000
        assert m["token-usage"]["output"] == 500
        assert "manifest.json" in m["files"]
        assert "created_at" in m

    def test_entry_fields(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        assert m["entry"]["build"] == "pip install -r requirements.txt"
        assert m["entry"]["test"] == "python -m pytest tests/ -v"
        # MUST use python src/prime.py, NOT python -m src.prime
        assert m["entry"]["start"] == "python src/prime.py"
        assert m["entry"]["health"] == "http://localhost:8401/health"

    def test_created_at_is_iso(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        # ISO-8601: contains 'T' separator
        assert "T" in m["created_at"]

    def test_has_three_contracts(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        contracts = m["contracts"]
        assert len(contracts) == 3
        names = {c["name"] for c in contracts}
        assert names == {"health-liveness", "stats-generation", "stats-schema"}

    def test_stats_generation_contract_has_generation_number(self, tmp_artifact: Path):
        m = build_manifest(
            generation=5,
            parent_generation=4,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        gen_contract = next(c for c in m["contracts"] if c["name"] == "stats-generation")
        assert gen_contract["expect"]["body_contains"]["generation"] == 5

    def test_manifest_json_in_files_list(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        assert "manifest.json" in m["files"]

    def test_artifact_hash_excludes_manifest(self, tmp_artifact: Path):
        """artifact-hash must be the same whether manifest.json is in file list or not."""
        m1 = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        m2 = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py", "manifest.json"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        assert m1["artifact-hash"] == m2["artifact-hash"]


class TestWriteManifest:
    def test_writes_valid_json(self, tmp_artifact: Path):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
        )
        write_manifest(tmp_artifact, m)
        written = json.loads((tmp_artifact / "manifest.json").read_text())
        assert written["generation"] == 1
        assert written["cambrian-version"] == 1
