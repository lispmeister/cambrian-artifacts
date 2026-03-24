"""Pydantic models for Prime I/O boundaries."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    input: int
    output: int


class Entry(BaseModel):
    build: str
    test: str
    start: str
    health: str


class Contract(BaseModel):
    name: str
    type: str
    method: str
    path: str
    expect: dict[str, Any]


class Manifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cambrian_version: int = Field(alias="cambrian-version", default=1)
    generation: int
    parent_generation: int = Field(alias="parent-generation")
    spec_hash: str = Field(alias="spec-hash")
    artifact_hash: str = Field(alias="artifact-hash")
    producer_model: str = Field(alias="producer-model")
    token_usage: TokenUsage = Field(alias="token-usage")
    files: list[str]
    created_at: str
    entry: Entry
    contracts: list[Contract] = Field(default_factory=list)


class GenerationRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    generation: int
    parent: int
    spec_hash: str = Field(default="", alias="spec-hash")
    artifact_hash: str = Field(default="", alias="artifact-hash")
    outcome: str
    artifact_ref: str = ""
    created: str | None = None
    completed: str | None = None
    container_id: str = Field(default="", alias="container-id")
    viability: dict[str, Any] | None = None
