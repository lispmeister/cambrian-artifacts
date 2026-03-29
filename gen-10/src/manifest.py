"""Manifest building and hash computation for Prime artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 hash of the spec file with sha256: prefix."""
    content = spec_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_spec_hash_from_content(content: bytes) -> str:
    """Compute SHA-256 hash from spec content bytes."""
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute SHA-256 hash of all artifact files except manifest.json.
    Files are processed in lexicographic sort order.
    Path and content are separated by a null byte.
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
    """
    pattern = r'```contracts\s*\n(.*?)\n```'
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
    producer_model: str,
    token_usage: dict[str, int],
    files: list[str],
    spec_content: str = "",
) -> dict[str, Any]:
    """Build the manifest dictionary with all required fields."""
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
        "files": files,
        "created_at": created_at,
        "entry": {
            "build": "pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python src/prime.py",
            "health": "http://localhost:8401/health",
        },
    }

    contracts = extract_contracts_from_spec(spec_content) if spec_content else None
    if contracts is not None:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest.json to the artifact root directory."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path