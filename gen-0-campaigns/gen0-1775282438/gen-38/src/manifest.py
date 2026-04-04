"""Manifest building, hash computation, and spec contract extraction."""
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
    Files are processed in lexicographic order with null-byte separator.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_content: str) -> list[dict[str, Any]]:
    """
    Extract JSON contracts array from a fenced code block marked with 'contracts'
    in the spec content.
    """
    pattern = r"```contracts\s*\n(.*?)\n```"
    match = re.search(pattern, spec_content, re.DOTALL)
    if not match:
        return []
    try:
        contracts = json.loads(match.group(1))
        if isinstance(contracts, list):
            return contracts
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def build_manifest(
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_hash: str,
    producer_model: str,
    token_usage: dict[str, int],
    files: list[str],
    spec_file_path: str,
    contracts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": token_usage,
        "files": files,
        "created-at": now,
        "entry": {
            "build": "uv pip install -r requirements.txt",
            "test": "python -m pytest tests/ -v",
            "start": "python -m src.prime",
            "health": "http://localhost:8401/health",
        },
    }

    if contracts:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest_data: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2))
