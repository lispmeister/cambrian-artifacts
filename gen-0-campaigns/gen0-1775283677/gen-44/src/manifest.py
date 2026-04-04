"""Manifest building, hash computation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file with sha256: prefix."""
    content = spec_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of all artifact files except manifest.json.
    Uses lexicographic sort and null byte separator between path and content.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):  # lexicographic sort required
        if rel_path == "manifest.json":
            continue  # manifest.json is excluded
        hasher.update(rel_path.encode())
        hasher.update(b"\0")  # null separator between path and content
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_content: str) -> list[dict[str, Any]] | None:
    """
    Extract contracts JSON array from spec content.
    Looks for a fenced code block marked with 'contracts'.
    Returns None if not found or invalid JSON.
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
    generation: int,
    parent_generation: int,
    spec_hash: str,
    files: list[str],
    producer_model: str,
    token_usage: dict[str, int],
    contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the manifest dict."""
    artifact_hash = compute_artifact_hash(artifact_root, files)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": {
            "input": token_usage.get("input", 0),
            "output": token_usage.get("output", 0),
        },
        "files": sorted(files),
        "created-at": created_at,
        "entry": {
            "build": "uv pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python -m src.prime",
            "health": "http://localhost:8401/health",
        },
    }

    if contracts is not None:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
