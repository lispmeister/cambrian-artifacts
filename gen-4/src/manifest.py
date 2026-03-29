"""Manifest building and hash computation."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute SHA-256 of spec file."""
    data = spec_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute artifact hash over all files except manifest.json.
    Files are sorted lexicographically. Path and content separated by null byte.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def build_manifest(
    *,
    generation: int,
    parent_generation: int,
    spec_hash: str,
    artifact_root: Path,
    files: list[str],
    producer_model: str,
    token_input: int,
    token_output: int,
    contracts: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the manifest dict (without artifact-hash, computed after)."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    artifact_hash = compute_artifact_hash(artifact_root, files)

    manifest: dict[str, Any] = {
        "cambrian-version": 1,
        "generation": generation,
        "parent-generation": parent_generation,
        "spec-hash": spec_hash,
        "artifact-hash": artifact_hash,
        "producer-model": producer_model,
        "token-usage": {
            "input": token_input,
            "output": token_output,
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

    if contracts is not None:
        manifest["contracts"] = contracts

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def extract_contracts_from_spec(spec_content: str) -> list[dict[str, Any]] | None:
    """
    Extract contracts from spec if present in a fenced code block marked 'contracts'.
    Returns None if not found.
    """
    pattern = r"```contracts\s*\n([\s\S]*?)\n```"
    m = re.search(pattern, spec_content)
    if m:
        try:
            return json.loads(m.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None
    return None