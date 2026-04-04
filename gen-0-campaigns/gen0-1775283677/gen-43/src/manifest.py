"""Manifest building, hash computation."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def compute_spec_hash(spec_path: Path) -> str:
    """Compute the SHA-256 hash of the spec file."""
    content = spec_path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    return f"sha256:{digest}"


def compute_artifact_hash(artifact_root: Path, files: list[str]) -> str:
    """
    Compute the SHA-256 hash of all artifact files except manifest.json.
    Files are sorted lexicographically and null-separated from content.
    """
    hasher = hashlib.sha256()
    for rel_path in sorted(files):
        if rel_path == "manifest.json":
            continue
        hasher.update(rel_path.encode())
        hasher.update(b"\0")
        hasher.update((artifact_root / rel_path).read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def extract_contracts_from_spec(spec_path: Path) -> list[dict[str, Any]] | None:
    """Extract contracts JSON array from spec file if present."""
    try:
        content = spec_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None

    lines = content.splitlines()
    in_contracts_block = False
    json_lines: list[str] = []

    for line in lines:
        if line.strip() == "```contracts":
            in_contracts_block = True
            json_lines = []
            continue
        if in_contracts_block:
            if line.strip() == "```":
                break
            json_lines.append(line)

    if not json_lines:
        return None

    try:
        result = json.loads("\n".join(json_lines))
        if isinstance(result, list):
            return result
        return None
    except json.JSONDecodeError:
        return None


def build_manifest(
    artifact_root: Path,
    files: list[str],
    generation: int,
    parent_generation: int,
    spec_hash: str,
    producer_model: str,
    token_usage: dict[str, int],
    spec_path: Path,
) -> dict[str, Any]:
    """Build the manifest dictionary."""
    artifact_hash = compute_artifact_hash(artifact_root, files)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Find the spec file relative path in files list
    spec_rel = None
    for f in files:
        if f.endswith(".md") and "spec" in f.lower():
            spec_rel = f
            break

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

    # Try to extract contracts from spec
    spec_contracts = extract_contracts_from_spec(spec_path)
    if spec_contracts is not None:
        manifest["contracts"] = spec_contracts

    return manifest


def write_manifest(artifact_root: Path, manifest_data: dict[str, Any]) -> None:
    """Write manifest.json to the artifact root."""
    manifest_path = artifact_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )
