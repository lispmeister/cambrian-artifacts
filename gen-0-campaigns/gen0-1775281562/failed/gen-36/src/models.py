"""Pydantic models for manifest, viability report, and generation records."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EntryPoints(BaseModel):
    build: str
    test: str
    start: str
    health: str


class TokenUsage(BaseModel):
    input: int
    output: int


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


class CheckResult(BaseModel):
    passed: bool
    duration_ms: int = 0


class ViabilityReport(BaseModel):
    generation: int
    status: str
    failure_stage: str
    checks: dict[str, Any]
    completed_at: str
    diagnostics: dict[str, Any] | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash", default="")
    outcome: str
    viability: ViabilityReport | None = None
    created: str
    completed: str | None = None
    container_id: str = Field(alias="container-id")
    campaign_id: str | None = Field(alias="campaign-id", default=None)

    model_config = {"populate_by_name": True}
