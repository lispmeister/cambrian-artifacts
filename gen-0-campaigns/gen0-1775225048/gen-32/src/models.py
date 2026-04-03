"""Pydantic models for manifest, viability report, and generation record."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input: int
    output: int


class EntryPoints(BaseModel):
    build: str
    test: str
    start: str
    health: str


class Manifest(BaseModel):
    cambrian_version: int = Field(alias="cambrian-version")
    generation: int
    parent_generation: int = Field(alias="parent-generation")
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    producer_model: str = Field(alias="producer-model")
    token_usage: TokenUsage = Field(alias="token-usage")
    files: list[str]
    created_at: str = Field(alias="created-at")
    entry: EntryPoints
    contracts: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ViabilityCheck(BaseModel):
    passed: bool
    duration_ms: int = 0


class ViabilityChecks(BaseModel):
    manifest: ViabilityCheck
    build: ViabilityCheck
    test: ViabilityCheck
    start: ViabilityCheck
    health: ViabilityCheck


class ViabilityDiagnostics(BaseModel):
    stage: str
    summary: str
    exit_code: int | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class ViabilityReport(BaseModel):
    generation: int
    status: str
    failure_stage: str
    checks: dict[str, Any]
    completed_at: str
    diagnostics: ViabilityDiagnostics | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    outcome: str
    created: str
    container_id: str = Field(alias="container-id")
    viability: ViabilityReport | None = None
    artifact_ref: str | None = Field(default=None, alias="artifact-ref")
    completed: str | None = None
    campaign_id: str | None = Field(default=None, alias="campaign-id")

    model_config = {"populate_by_name": True}
