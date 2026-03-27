"""Tests for manifest building and hash computation."""
import hashlib
import json
from pathlib import Path

import pytest
from src.manifest import (
    compute_spec_hash,
    compute_artifact_hash,
    extract_contracts,
    build_manifest,
)


@pytest.fixture
def tmp_artifact(tmp_path):
    """Create a minimal artifact directory."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')")
    (tmp_path / "requirements.txt").write_text("aiohttp>=3.9\n")
    return tmp_path


class TestComputeSpecHash:
    def test_matches_sha256(self, tmp_path):
        spec = tmp_path / "spec.md"
        spec.write_text("# My Spec")
        expected = "sha256:" + hashlib.sha256(b"# My Spec").hexdigest()
        assert compute_spec_hash(spec) == expected


class TestComputeArtifactHash:
    def test_excludes_manifest(self, tmp_artifact):
        (tmp_artifact / "manifest.json").write_text("{}")
        files = ["src/main.py", "requirements.txt", "manifest.json"]
        h1 = compute_artifact_hash(tmp_artifact, files)
        h2 = compute_artifact_hash(tmp_artifact, ["src/main.py", "requirements.txt"])
        assert h1 == h2

    def test_deterministic(self, tmp_artifact):
        files = ["src/main.py", "requirements.txt"]
        h1 = compute_artifact_hash(tmp_artifact, files)
        h2 = compute_artifact_hash(tmp_artifact, files)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_artifact):
        files = ["src/main.py", "requirements.txt"]
        h1 = compute_artifact_hash(tmp_artifact, files)
        (tmp_artifact / "src" / "main.py").write_text("print('different')")
        h2 = compute_artifact_hash(tmp_artifact, files)
        assert h1 != h2

    def test_includes_file_paths_in_hash(self, tmp_artifact):
        """Renaming a file changes the hash even if content is identical."""
        (tmp_artifact / "src" / "other.py").write_text("print('hello')")
        h1 = compute_artifact_hash(tmp_artifact, ["src/main.py"])
        h2 = compute_artifact_hash(tmp_artifact, ["src/other.py"])
        assert h1 != h2, "Hash should differ when file path differs, even with same content"

    def test_null_separator_prevents_collision(self, tmp_artifact):
        """Null-byte separator prevents path/content boundary collisions."""
        # Without separator: hash("a_file" + "hello") == hash("a_fil" + "ehello")
        # With separator:    hash("a_file\0hello")    != hash("a_fil\0ehello")
        (tmp_artifact / "src" / "a_file.py").write_text("hello")
        (tmp_artifact / "src" / "a_fil.py").write_text("ehello")
        h1 = compute_artifact_hash(tmp_artifact, ["src/a_file.py"])
        h2 = compute_artifact_hash(tmp_artifact, ["src/a_fil.py"])
        assert h1 != h2, "Null-byte separator must prevent path/content boundary collisions"

    def test_matches_spec_algorithm(self, tmp_artifact):
        """Hash output matches the spec's reference algorithm exactly."""
        files = ["src/main.py", "requirements.txt"]
        result = compute_artifact_hash(tmp_artifact, files)

        # Recompute manually per spec
        hasher = hashlib.sha256()
        for f in sorted(files):
            hasher.update(f.encode())
            hasher.update(b"\0")
            hasher.update((tmp_artifact / f).read_bytes())
        expected = "sha256:" + hasher.hexdigest()

        assert result == expected


class TestExtractContracts:
    def test_returns_none_when_no_block(self):
        assert extract_contracts("# No contracts here") is None

    def test_extracts_valid_contracts(self):
        spec = '```contracts\n[{"name": "test", "type": "http"}]\n```'
        result = extract_contracts(spec)
        assert result == [{"name": "test", "type": "http"}]

    def test_returns_none_on_invalid_json(self):
        spec = "```contracts\nnot valid json\n```"
        assert extract_contracts(spec) is None

    def test_returns_none_if_not_a_list(self):
        spec = '```contracts\n{"not": "a list"}\n```'
        assert extract_contracts(spec) is None


class TestBuildManifest:
    def test_has_all_required_fields(self, tmp_artifact):
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
        assert m["entry"]["build"] == "pip install -r requirements.txt"
        assert m["entry"]["test"] == "python -m pytest tests/ -v"
        assert m["entry"]["start"] == "python -m src.prime"
        assert m["entry"]["health"] == "http://localhost:8401/health"

    def test_created_at_is_iso(self, tmp_artifact):
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
        assert "T" in m["created_at"]

    def test_has_three_contracts_by_default(self, tmp_artifact):
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

    def test_stats_generation_contract_has_generation(self, tmp_artifact):
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
        gen_contract = [c for c in m["contracts"] if c["name"] == "stats-generation"][0]
        assert gen_contract["expect"]["body_contains"]["generation"] == 5

    def test_uses_contracts_from_spec_when_provided(self, tmp_artifact):
        spec_with_contracts = (
            '```contracts\n'
            '[{"name": "custom", "type": "http", "method": "GET", "path": "/custom",'
            ' "expect": {"status": 200}}]\n'
            '```'
        )
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
            spec_content=spec_with_contracts,
        )
        assert len(m["contracts"]) == 1
        assert m["contracts"][0]["name"] == "custom"

    def test_falls_back_to_defaults_when_spec_has_no_contracts_block(self, tmp_artifact):
        m = build_manifest(
            generation=1,
            parent_generation=0,
            spec_hash="sha256:abc",
            artifact_dir=tmp_artifact,
            files=["src/main.py"],
            producer_model="test",
            input_tokens=0,
            output_tokens=0,
            spec_content="# No contracts block here",
        )
        assert len(m["contracts"]) == 3
