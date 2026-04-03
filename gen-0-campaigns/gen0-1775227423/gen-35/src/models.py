"""Pydantic models for Cambrian Prime."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0


class ManifestEntry(BaseModel):
    build: str
    test: str
    start: str
    health: str


class Manifest(BaseModel):
    cambrian_version: int = Field(alias="cambrian-version", default=1)
    generation: int
    parent_generation: int = Field(alias="parent-generation")
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    producer_model: str = Field(alias="producer-model")
    token_usage: TokenUsage = Field(alias="token-usage")
    files: list[str]
    created_at: str = Field(alias="created-at")
    entry: ManifestEntry
    contracts: list[dict[str, Any]] | None = None

    model_config = {"populate_by_name": True}


class DiagnosticsInfo(BaseModel):
    stage: str = ""
    summary: str = ""
    exit_code: int | None = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class CheckResult(BaseModel):
    passed: bool = False
    duration_ms: int = 0
    tests_run: int | None = None
    tests_passed: int | None = None
    contracts: dict[str, Any] | None = None


class ViabilityChecks(BaseModel):
    manifest: CheckResult = Field(default_factory=CheckResult)
    build: CheckResult = Field(default_factory=CheckResult)
    test: CheckResult = Field(default_factory=CheckResult)
    start: CheckResult = Field(default_factory=CheckResult)
    health: CheckResult = Field(default_factory=CheckResult)


class ViabilityReport(BaseModel):
    generation: int = 0
    status: str = "non-viable"
    failure_stage: str = "none"
    checks: ViabilityChecks = Field(default_factory=ViabilityChecks)
    completed_at: str = ""
    diagnostics: DiagnosticsInfo | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int = 0
    spec_hash: str = Field(alias="spec-hash", default="")
    artifact_hash: str = Field(alias="artifact-hash", default="")
    outcome: str = "in_progress"
    viability: ViabilityReport | None = None
    artifact_ref: str | None = Field(alias="artifact-ref", default=None)
    created: str = ""
    completed: str | None = None
    container_id: str = Field(alias="container-id", default="")
    campaign_id: str | None = Field(alias="campaign-id", default=None)

    model_config = {"populate_by_name": True}
