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
    Compute SHA-256 hash of all artifact files except manifest.json.
    Uses lexicographic sort and null byte separator between path and content.
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
    Extract contracts from spec content.
    Looks for a JSON array in a fenced code block marked with 'contracts'.
    """
    import re
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
    generation: int,
    parent_generation: int,
    spec_path: Path,
    artifact_root: Path,
    files: list[str],
    model: str,
    token_usage: dict[str, int],
    spec_content: str,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    spec_hash = compute_spec_hash(spec_path)
    artifact_hash = compute_artifact_hash(artifact_root, files)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

    contracts = extract_contracts_from_spec(spec_content)
    if contracts is not None:
        manifest["contracts"] = contracts
    else:
        manifest["contracts"] = [
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
        ]

    return manifest


def write_manifest(artifact_root: Path, manifest: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
