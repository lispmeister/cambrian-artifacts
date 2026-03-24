"""Tests for manifest building and hash computation."""
import hashlib
import json
from pathlib import Path

import pytest
from src.manifest import compute_spec_hash, compute_artifact_hash, build_manifest


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
        # ISO-8601: contains 'T' separator
        assert "T" in m["created_at"]
