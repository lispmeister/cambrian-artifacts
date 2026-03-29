#!/usr/bin/env python3
"""Manifest building, hash computation."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file."""
    data = spec_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of all artifact files except manifest.json.
    Uses lexicographic sort and null separator between path and content.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_content: str) -> list[dict[str, Any]] | None:
    """
    Extract contracts JSON array from spec content.
    Looks for a fenced code block marked with 'contracts'.
    Returns None if not found or invalid JSON.
    """
    pattern = r"```contracts\s*\n(.*?)\n```"
    match = re.search(pattern, spec_content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        if isinstance(data, list):
            return data
        return None
    except json.JSONDecodeError:
        return None


def build_manifest(
    artifact_root: Path,
    files: list[str],
    generation: int,
    parent_generation: int,
    spec_hash: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict (without writing it)."""
    artifact_hash = compute_artifact_hash(artifact_root, files)

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": model,
        "token-usage": {
            "input": input_tokens,
            "output": output_tokens,
        },
        "files": sorted(files),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry": {
            "build": "pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python src/prime.py",
            "health": "http://localhost:8401/health",
        },
        "contracts": contracts if contracts is not None else [
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
                "expect": {"status": 200, "body_contains": {"generation": "$GENERATION"}},
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


def write_manifest(artifact_root: Path, manifest_data: dict[str, Any]) -> Path:
    """Write manifest.json to artifact root. Returns path."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_path