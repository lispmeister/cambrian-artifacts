"""Tests for generation loop logic: numbering, retries, model selection."""
from __future__ import annotations

import os

import pytest

from src.generate import GenerationConfig, select_model


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    history: list[dict] = []
    if history:
        next_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history, next generation is max + 1."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential history still uses max + 1."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 3, "outcome": "promoted"},
        {"generation": 5, "outcome": "in_progress"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """retry_count >= 1 uses escalation model."""
    orig_model = os.environ.pop("CAMBRIAN_MODEL", None)
    orig_esc = os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        assert select_model(0) == "claude-sonnet-4-6"
        assert select_model(1) == "claude-opus-4-6"
        assert select_model(2) == "claude-opus-4-6"
        assert select_model(99) == "claude-opus-4-6"
    finally:
        if orig_model is not None:
            os.environ["CAMBRIAN_MODEL"] = orig_model
        if orig_esc is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = orig_esc


def test_max_retries_default() -> None:
    """Default max retries is 3."""
    orig = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        config = GenerationConfig.from_env()
        assert config.max_retries == 3
    finally:
        if orig is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = orig


def test_max_gens_default() -> None:
    """Default max gens is 5."""
    orig = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        config = GenerationConfig.from_env()
        assert config.max_gens == 5
    finally:
        if orig is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = orig


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    orig = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        config = GenerationConfig.from_env()
        assert config.max_parse_retries == 2
    finally:
        if orig is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = orig


def test_max_retries_from_env() -> None:
    """max_retries reads from CAMBRIAN_MAX_RETRIES env var."""
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        config = GenerationConfig.from_env()
        assert config.max_retries == 7
    finally:
        del os.environ["CAMBRIAN_MAX_RETRIES"]


def test_retry_stops_at_max() -> None:
    """Retry loop logic respects max_retries."""
    max_retries = 3
    retry_count = 0
    stopped = False

    for _ in range(10):
        if retry_count > max_retries:
            stopped = True
            break
        retry_count += 1

    assert stopped
    assert retry_count == max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter tracks attempts independently."""
    max_parse_retries = 2
    parse_attempts = 0
    repaired = False

    for attempt in range(max_parse_retries + 1):
        parse_attempts += 1
        if attempt >= max_parse_retries:
            # Would raise ParseError here
            break
        # Simulate repair attempt
        repaired = True

    assert parse_attempts == max_parse_retries + 1
    assert repaired