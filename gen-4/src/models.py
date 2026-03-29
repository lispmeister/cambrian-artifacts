"""Pydantic models for Prime: manifest, viability report, generation records."""

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


class Contract(BaseModel):
    name: str
    type: str
    method: str
    path: str
    expect: dict[str, Any]


class Manifest(BaseModel):
    cambrian_version: int = Field(alias="cambrian-version")
    generation: int
    parent_generation: int = Field(alias="parent-generation")
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    producer_model: str = Field(alias="producer-model")
    token_usage: TokenUsage = Field(alias="token-usage")
    files: list[str]
    created_at: str
    entry: EntryPoints
    contracts: list[Contract] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class CheckResult(BaseModel):
    passed: bool
    duration_ms: float = 0
    tests_run: int | None = None
    tests_passed: int | None = None


class ViabilityFailure(BaseModel):
    test: str = ""
    error: str = ""
    file: str = ""
    line: int = 0


class ViabilityDiagnostics(BaseModel):
    stage: str
    summary: str
    exit_code: int
    failures: list[ViabilityFailure] = Field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


class ViabilityReport(BaseModel):
    generation: int
    status: str  # "viable" | "non-viable"
    failure_stage: str
    checks: dict[str, CheckResult]
    completed_at: str
    diagnostics: ViabilityDiagnostics | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    outcome: str  # in_progress, tested, promoted, failed, timeout
    viability: ViabilityReport | None = None
    artifact_ref: str | None = None
    created: str
    completed: str | None = None
    container_id: str = Field(alias="container-id")

    model_config = {"populate_by_name": True}


class SpawnRequest(BaseModel):
    spec_hash: str = Field(alias="spec-hash")
    generation: int
    artifact_path: str = Field(alias="artifact-path")

    model_config = {"populate_by_name": True}


class SpawnResponse(BaseModel):
    ok: bool
    container_id: str | None = Field(default=None, alias="container-id")
    generation: int | None = None
    error: str | None = None

    model_config = {"populate_by_name": True}


class PromoteRollbackResponse(BaseModel):
    ok: bool
    generation: int | None = None
    error: str | None = None