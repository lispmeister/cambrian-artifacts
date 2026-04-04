"""Pydantic models for manifest, viability report, and generation records."""
from __future__ import annotations

from typing import Any, Literal

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
    contracts: list[dict[str, Any]] | None = None

    model_config = {"populate_by_name": True}


class CheckResult(BaseModel):
    passed: bool
    duration_ms: int | None = None


class TestCheckResult(CheckResult):
    tests_run: int | None = None
    tests_passed: int | None = None


class ViabilityChecks(BaseModel):
    manifest: CheckResult | None = None
    build: CheckResult | None = None
    test: TestCheckResult | None = None
    start: CheckResult | None = None
    health: CheckResult | None = None


class ViabilityFailure(BaseModel):
    test: str | None = None
    error: str | None = None
    file: str | None = None
    line: int | None = None


class ViabilityDiagnostics(BaseModel):
    stage: str
    summary: str
    exit_code: int | None = None
    failures: list[ViabilityFailure] = []
    stdout_tail: str = ""
    stderr_tail: str = ""


class ViabilityReport(BaseModel):
    generation: int
    status: Literal["viable", "non-viable"]
    failure_stage: str
    checks: ViabilityChecks
    completed_at: str
    diagnostics: ViabilityDiagnostics | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    outcome: str
    viability: ViabilityReport | None = None
    artifact_ref: str | None = Field(default=None, alias="artifact-ref")
    created: str
    completed: str | None = None
    container_id: str = Field(alias="container-id")
    campaign_id: str | None = Field(default=None, alias="campaign-id")

    model_config = {"populate_by_name": True}
