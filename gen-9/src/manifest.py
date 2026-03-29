#!/usr/bin/env python3
"""Manifest building, hash computation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file with sha256: prefix."""
    data = spec_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of artifact files (excluding manifest.json).

    Files are sorted lexicographically. Path and content are separated by
    a null byte to prevent hash collisions.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_content: str) -> Optional[list[dict[str, Any]]]:
    """
    Extract contracts JSON array from spec content.

    Looks for a fenced code block marked with 'contracts' language identifier.
    Returns the parsed JSON array, or None if not found or invalid.
    """
    pattern = r"```contracts\s*\n([\s\S]*?)\n```"
    match = re.search(pattern, spec_content)
    if not match:
        return None
    try:
        contracts = json.loads(match.group(1))
        if isinstance(contracts, list):
            return contracts
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def build_manifest(
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_root: Path,
    files: list[str],
    model: str,
    token_usage: dict[str, int],
    contracts: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Build the manifest dict for an artifact."""
    # Ensure manifest.json is in files list
    all_files = list(files)
    if "manifest.json" not in all_files:
        all_files.append("manifest.json")

    artifact_hash = compute_artifact_hash(artifact_root, all_files)

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
        "files": all_files,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entry": {
            "build": "pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python src/prime.py",
            "health": "http://localhost:8401/health",
        },
    }

    if contracts is not None:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json to the artifact root. Returns the path."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest_path