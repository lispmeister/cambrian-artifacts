"""Manifest building, hash computation."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of spec file with sha256: prefix."""
    data = spec_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of artifact files (excluding manifest.json).
    Files are sorted lexicographically. Each entry: path_bytes + null_byte + content_bytes.
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
    Extract contracts JSON array from spec if present in a fenced code block
    marked with 'contracts'.
    Returns None if not found or invalid.
    """
    pattern = r"```contracts\s*\n([\s\S]*?)```"
    match = re.search(pattern, spec_content)
    if not match:
        return None
    try:
        data = json.loads(match.group(1).strip())
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
    spec_hash: str,
    producer_model: str,
    token_usage: dict[str, int],
    spec_content: str = "",
) -> dict[str, Any]:
    """Build the manifest dict (without writing it)."""
    # Exclude manifest.json from artifact hash computation
    artifact_hash = compute_artifact_hash(artifact_root, files)

    # Include manifest.json in file list if not already present
    all_files = list(files)
    if "manifest.json" not in all_files:
        all_files.append("manifest.json")

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": token_usage,
        "files": sorted(all_files),
        "created-at": created_at,
        "entry": {
            "build": "uv pip install -r requirements.txt",
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

    # Override contracts from spec if present
    if spec_content:
        spec_contracts = extract_contracts_from_spec(spec_content)
        if spec_contracts is not None:
            manifest["contracts"] = spec_contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
