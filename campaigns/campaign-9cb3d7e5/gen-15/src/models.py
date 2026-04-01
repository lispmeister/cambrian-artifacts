#!/usr/bin/env python3
"""Pydantic models for manifest, viability report, generation record."""

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


class ManifestModel(BaseModel):
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


class TestCheckResult(BaseModel):
    passed: bool
    duration_ms: int = 0
    tests_run: int = 0
    tests_passed: int = 0


class HealthCheckResult(BaseModel):
    passed: bool
    duration_ms: int = 0
    contracts: dict[str, Any] = Field(default_factory=dict)


class Checks(BaseModel):
    manifest: CheckResult
    build: CheckResult
    test: TestCheckResult
    start: CheckResult
    health: HealthCheckResult


class Diagnostics(BaseModel):
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
    checks: Checks
    completed_at: str
    diagnostics: Diagnostics | None = None


class GenerationRecord(BaseModel):
    generation: int
    parent: int
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash", default="")
    outcome: str
    viability: ViabilityReport | None = None
    artifact_ref: str | None = Field(alias="artifact-ref", default=None)
    created: str
    completed: str | None = None
    container_id: str = Field(alias="container-id")

    model_config = {"populate_by_name": True}
