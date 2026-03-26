"""Manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """SHA-256 hash of the spec file, with sha256: prefix."""
    return "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()


def compute_artifact_hash(artifact_dir: Path, files: list[str]) -> str:
    """SHA-256 hash of all artifact files except manifest.json, sorted lexicographically.

    Uses a null-byte separator between the file path and file content so that
    complementary path/content splits (e.g. path="ab" content="c" vs path="a" content="bc")
    cannot collide.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):          # lexicographic sort is required
        if rel_path == "manifest.json":
            continue                        # manifest.json is excluded
        file_path = artifact_dir / rel_path
        if file_path.exists():
            hasher.update(rel_path.encode())
            hasher.update(b"\0")            # null separator between path and content
            hasher.update(file_path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def build_manifest(
    *,
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_dir: Path,
    files: list[str],
    producer_model: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Build a complete manifest dict for the artifact."""
    # Ensure manifest.json is in the file list for the final manifest
    all_files = sorted(set(files) | {"manifest.json"})
    # Compute artifact hash over files (excluding manifest.json per algorithm)
    artifact_hash = compute_artifact_hash(artifact_dir, all_files)

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": {"input": input_tokens, "output": output_tokens},
        "files": all_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry": {
            "build": "pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python src/prime.py",
            "health": "http://localhost:8401/health",
        },
        "contracts": [
            {
                "name": "health-liveness",
                "type": "http",
                "method": "GET",
                "path": "/health",
                "expect": {"status": 200, "body": {"ok": True}},
            },
            {
                "name": "stats-generation",
                "type": "http",
                "method": "GET",
                "path": "/stats",
                "expect": {"status": 200, "body_contains": {"generation": generation}},
            },
            {
                "name": "stats-schema",
                "type": "http",
                "method": "GET",
                "path": "/stats",
                "expect": {"status": 200, "body_has_keys": ["generation", "status", "uptime"]},
            },
        ],
    }
    return manifest


def write_manifest(artifact_dir: Path, manifest_data: dict[str, Any]) -> None:
    """Write manifest.json to the artifact directory."""
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2))
