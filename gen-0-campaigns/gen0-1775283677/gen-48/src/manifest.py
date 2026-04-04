"""Manifest building and hash computation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file."""
    content = spec_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of artifact files (excluding manifest.json).
    Files are sorted lexicographically. Each file contributes its path bytes,
    a null separator, and then its content bytes.
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
    Extract contracts from a JSON array in a fenced code block marked 'contracts'
    in the spec content. Returns None if not found or invalid.
    """
    import re
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
    artifact_root: Path,
    files: list[str],
    generation: int,
    parent_generation: int,
    spec_hash: str,
    spec_content: str,
    token_usage: dict[str, int],
    model: str,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    artifact_hash = compute_artifact_hash(artifact_root, files)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Ensure manifest.json is in the files list
    all_files = list(files)
    if "manifest.json" not in all_files:
        all_files.append("manifest.json")

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

    # Override contracts if spec defines them
    spec_contracts = extract_contracts_from_spec(spec_content)
    if spec_contracts is not None:
        manifest["contracts"] = spec_contracts

    return manifest


def write_manifest(artifact_root: Path, manifest_data: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
