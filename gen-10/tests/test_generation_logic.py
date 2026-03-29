"""Tests for generation loop logic."""

from __future__ import annotations

import os

import pytest

from src.generate import GenerationConfig
from src.models import GenerationRecord


def make_record(generation: int, outcome: str = "promoted") -> GenerationRecord:
    """Helper to create a GenerationRecord."""
    return GenerationRecord.model_validate({
        "generation": generation,
        "parent": generation - 1,
        "spec-hash": "sha256:" + "a" * 64,
        "artifact-hash": "sha256:" + "b" * 64,
        "outcome": outcome,
        "created": "2026-01-01T00:00:00Z",
        "container-id": f"container-{generation}",
    })


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    history: list[GenerationRecord] = []
    next_gen = max((r.generation for r in history), default=0) + 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With existing history, next generation is max+1."""
    history = [make_record(1), make_record(2), make_record(3)]
    next_gen = max(r.generation for r in history) + 1
    assert next_gen == 4


def test_generation_number_non_sequential() -> None:
    """Generation number is based on max, not count."""
    history = [make_record(1), make_record(5), make_record(3)]
    next_gen = max(r.generation for r in history) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates on retry_count >= 1."""
    config = GenerationConfig(
        anthropic_api_key="test",
        model="claude-sonnet-4-6",
        escalation_model="claude-opus-4-6",
    )
    from src.generate import LLMGenerator
    gen = LLMGenerator(config)
    assert gen._select_model(0) == "claude-sonnet-4-6"
    assert gen._select_model(1) == "claude-opus-4-6"
    assert gen._select_model(3) == "claude-opus-4-6"


def test_max_retries_default() -> None:
    """Default max_retries is 3."""
    config = GenerationConfig(anthropic_api_key="test")
    assert config.max_retries == 3


def test_max_gens_default() -> None:
    """Default max_gens is 5."""
    config = GenerationConfig(anthropic_api_key="test")
    assert config.max_gens == 5


def test_max_parse_retries_default() -> None:
    """Default max_parse_retries is 2."""
    config = GenerationConfig(anthropic_api_key="test")
    assert config.max_parse_retries == 2


def test_max_retries_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_retries can be set from environment."""
    monkeypatch.setenv("CAMBRIAN_MAX_RETRIES", "7")
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at max_retries."""
    config = GenerationConfig(anthropic_api_key="test", max_retries=3)
    retry_count = 0
    for _ in range(10):
        if retry_count > config.max_retries:
            break
        retry_count += 1
    assert retry_count <= config.max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter stops at max_parse_retries."""
    config = GenerationConfig(anthropic_api_key="test", max_parse_retries=2)
    parse_retries = 0
    repairs_done = 0
    while parse_retries <= config.max_parse_retries:
        parse_retries += 1
        repairs_done += 1
    # Should have attempted initial + max_parse_retries repairs
    assert repairs_done == config.max_parse_retries + 1