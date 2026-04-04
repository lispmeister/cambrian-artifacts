"""Pydantic models for manifest, viability report, generation record."""
from __future__ import annotations

from typing import Any, Optional
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
    contracts: Optional[list[dict[str, Any]]] = None

    model_config = {"populate_by_name": True}


class CheckResult(BaseModel):
    passed: bool
    duration_ms: Optional[int] = None


class TestCheckResult(CheckResult):
    tests_run: Optional[int] = None
    tests_passed: Optional[int] = None


class ViabilityChecks(BaseModel):
    manifest: Optional[CheckResult] = None
    build: Optional[CheckResult] = None
    test: Optional[TestCheckResult] = None
    start: Optional[CheckResult] = None
    health: Optional[CheckResult] = None


class Diagnostics(BaseModel):
    stage: str
    summary: str
    exit_code: Optional[int] = None
    failures: list[dict[str, Any]] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class ViabilityReport(BaseModel):
    generation: int
    status: str
    failure_stage: str
    checks: ViabilityChecks
    completed_at: str
    diagnostics: Optional[Diagnostics] = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    outcome: str
    viability: Optional[ViabilityReport] = None
    artifact_ref: Optional[str] = Field(default=None, alias="artifact-ref")
    created: str
    completed: Optional[str] = None
    container_id: str = Field(alias="container-id")
    campaign_id: Optional[str] = Field(default=None, alias="campaign-id")

    model_config = {"populate_by_name": True}
