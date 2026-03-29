#!/usr/bin/env python3
"""Pydantic models for manifest, viability report, generation record."""
from __future__ import annotations

from datetime import datetime
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
    created_at: str
    entry: EntryPoints
    contracts: Optional[list[dict[str, Any]]] = None

    model_config = {"populate_by_name": True}


class CheckResult(BaseModel):
    passed: bool
    duration_ms: Optional[int] = None
    tests_run: Optional[int] = None
    tests_passed: Optional[int] = None


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
    created: str
    container_id: str = Field(alias="container-id")
    viability: Optional[ViabilityReport] = None
    artifact_ref: Optional[str] = None
    completed: Optional[str] = None

    model_config = {"populate_by_name": True}