"""Manifest building and hash computation."""
from __future__ import annotations

import hashlib
import json
import re
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
    Compute SHA-256 hash of all artifact files except manifest.json.
    Files are sorted lexicographically. Path and content are separated by a null byte.
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
    Extract the contracts JSON array from a fenced code block marked with 'contracts'
    in the spec content.
    """
    pattern = r"```contracts\s*\n(.*?)\n```"
    match = re.search(pattern, spec_content, re.DOTALL)
    if not match:
        return None
    try:
        contracts = json.loads(match.group(1))
        if isinstance(contracts, list):
            return contracts
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def build_manifest(
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_root: Path,
    files: list[str],
    model: str,
    token_usage: dict[str, int],
    contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    artifact_hash = compute_artifact_hash(artifact_root, files)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
