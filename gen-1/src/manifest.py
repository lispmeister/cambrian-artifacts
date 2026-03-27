"""Manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """SHA-256 hash of the spec file."""
    return "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest()


def compute_artifact_hash(artifact_dir: Path, files: list[str]) -> str:
    """SHA-256 hash of all artifact files except manifest.json, sorted by path.

    Includes both file paths and file contents in the hash so that renames
    are detected even when content is unchanged. A null-byte separator between
    path and content prevents hash collisions at path/content boundaries.
    """
    hasher = hashlib.sha256()
    for f in sorted(files):
        if f == "manifest.json":
            continue
        file_path = artifact_dir / f
        if file_path.exists():
            hasher.update(f.encode())
            hasher.update(b"\0")
            hasher.update(file_path.read_bytes())
    return "sha256:" + hasher.hexdigest()


def extract_contracts(spec_content: str) -> list[dict[str, Any]] | None:
    """Extract contracts JSON array from a ```contracts block in the spec.

    Returns None if no contracts block is found or the block is invalid JSON.
    """
    m = re.search(r"```contracts\s*\n(.*?)\n```", spec_content, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


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
    spec_content: str | None = None,
) -> dict[str, Any]:
    """Build a complete manifest dict for the artifact."""
    artifact_hash = compute_artifact_hash(artifact_dir, files)

    # Contracts: extract from spec if available, else use hardcoded defaults
    contracts: list[dict[str, Any]] | None = None
    if spec_content is not None:
        contracts = extract_contracts(spec_content)
    if contracts is None:
        contracts = [
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
        ]

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
        "contracts": contracts,
    }
    return manifest


def write_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact directory."""
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
