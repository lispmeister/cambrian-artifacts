#!/usr/bin/env python3
"""Manifest building, hash computation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file with sha256: prefix."""
    content = spec_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute artifact hash over all files except manifest.json.
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
    Extract contracts from a JSON array in a fenced code block marked 'contracts'.
    Returns None if not found or invalid.
    """
    pattern = r"```contracts\s*\n([\s\S]*?)\n```"
    m = re.search(pattern, spec_content)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        if isinstance(data, list):
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def build_manifest(
    artifact_root: Path,
    files: list[str],
    generation: int,
    parent_generation: int,
    spec_path: Path,
    model: str,
    token_usage: dict[str, int],
) -> dict[str, Any]:
    """Build the manifest dict with all required fields."""
    spec_hash = compute_spec_hash(spec_path)

    # Include manifest.json in files list
    all_files = list(files)
    if "manifest.json" not in all_files:
        all_files.append("manifest.json")

    artifact_hash = compute_artifact_hash(artifact_root, all_files)

    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": model,
        "token-usage": {
            "input": token_usage.get("input", 0),
            "output": token_usage.get("output", 0),
        },
        "files": sorted(all_files),
        "created-at": created_at,
        "entry": {
            "build": "uv pip install -r requirements.txt",
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

    # Try to extract contracts from spec if available
    try:
        spec_content = spec_path.read_text(encoding="utf-8")
        extracted = extract_contracts_from_spec(spec_content)
        if extracted is not None:
            manifest["contracts"] = extracted
    except Exception:
        pass

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
