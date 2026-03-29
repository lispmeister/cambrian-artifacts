"""Pydantic models for Prime — manifest, viability report, generation record."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EntryPoints(BaseModel):
    build: str
    test: str
    start: str
    health: str


class TokenUsage(BaseModel):
    input: int
    output: int


class ContractExpect(BaseModel):
    status: int
    body: Optional[dict[str, Any]] = None
    body_contains: Optional[dict[str, Any]] = None
    body_has_keys: Optional[list[str]] = None


class Contract(BaseModel):
    name: str
    type: str
    method: str
    path: str
    expect: ContractExpect


class Manifest(BaseModel):
    cambrian_version: int = Field(alias="cambrian-version")
    generation: int
    parent_generation: int = Field(alias="parent-generation")
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    producer_model: str
    token_usage: TokenUsage
    files: list[str]
    created_at: str
    entry: EntryPoints
    contracts: Optional[list[Contract]] = None

    model_config = {"populate_by_name": True}


class CheckResult(BaseModel):
    passed: bool
    duration_ms: Optional[int] = None
    tests_run: Optional[int] = None
    tests_passed: Optional[int] = None
    contracts: Optional[dict[str, Any]] = None


class ViabilityChecks(BaseModel):
    manifest: Optional[CheckResult] = None
    build: Optional[CheckResult] = None
    test: Optional[CheckResult] = None
    start: Optional[CheckResult] = None
    health: Optional[CheckResult] = None


class FailureInfo(BaseModel):
    test: Optional[str] = None
    error: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None


class Diagnostics(BaseModel):
    stage: str
    summary: str
    exit_code: int
    failures: list[FailureInfo] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class ViabilityReport(BaseModel):
    generation: int
    status: str  # "viable" or "non-viable"
    failure_stage: str
    checks: ViabilityChecks = Field(default_factory=ViabilityChecks)
    completed_at: str
    diagnostics: Optional[Diagnostics] = None
    fitness: Optional[dict[str, Any]] = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    outcome: str
    viability: Optional[ViabilityReport] = None
    artifact_ref: Optional[str] = None
    created: str
    completed: Optional[str] = None
    container_id: str = Field(alias="container-id")

    model_config = {"populate_by_name": True}


class SpawnRequest(BaseModel):
    spec_hash: str = Field(alias="spec-hash")
    generation: int
    artifact_path: str = Field(alias="artifact-path")

    model_config = {"populate_by_name": True}


class SpawnResponse(BaseModel):
    ok: bool
    container_id: Optional[str] = Field(default=None, alias="container-id")
    generation: Optional[int] = None
    error: Optional[str] = None

    model_config = {"populate_by_name": True}


class PromoteRollbackRequest(BaseModel):
    generation: int


class PromoteRollbackResponse(BaseModel):
    ok: bool
    generation: Optional[int] = None
    error: Optional[str] = None