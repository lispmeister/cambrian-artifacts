"""Manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """SHA-256 hash of the spec file."""
    return "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()


def compute_artifact_hash(artifact_dir: Path, files: list[str]) -> str:
    """SHA-256 hash of all artifact files except manifest.json, sorted by path.

    Includes both file paths and file contents in the hash so that renames
    are detected even when content is unchanged.
    """
    hasher = hashlib.sha256()
    for f in sorted(files):
        if f == "manifest.json":
            continue
        file_path = artifact_dir / f
        if file_path.exists():
            hasher.update(f.encode())
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
    artifact_hash = compute_artifact_hash(artifact_dir, files)

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": {"input": input_tokens, "output": output_tokens},
        "files": sorted(files + ["manifest.json"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry": {
            "build": "pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python -m src.prime",
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


def write_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact directory."""
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
