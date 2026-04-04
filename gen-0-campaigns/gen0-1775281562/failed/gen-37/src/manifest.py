"""Manifest building, hash computation."""
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
    Files are processed in lexicographic order with null separator between path and content.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        file_path = artifact_root / rel_path
        if not file_path.exists():
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update(file_path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_content: str) -> list[dict[str, Any]] | None:
    """
    Extract contracts from a fenced code block marked with 'contracts' in the spec.
    Returns None if not found or invalid JSON.
    """
    # Look for ```contracts ... ``` block
    pattern = r"```contracts\s*\n(.*?)\n```"
    match = re.search(pattern, spec_content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        if isinstance(data, list):
            return data
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def build_manifest(
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_hash: str,
    files: list[str],
    token_usage: dict[str, int],
    spec_path: Path,
    spec_content: str,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    import os

    model = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Determine spec file path in artifact
    spec_filename = spec_path.name
    spec_artifact_path = f"spec/{spec_filename}"

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

    # Extract contracts from spec if present
    contracts = extract_contracts_from_spec(spec_content)
    if contracts is not None:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
