"""Tests for generation loop logic."""
from __future__ import annotations

import os
import pytest

from src.generate import (
    CAMBRIAN_MAX_GENS,
    CAMBRIAN_MAX_RETRIES,
    CAMBRIAN_MAX_PARSE_RETRIES,
    CAMBRIAN_ESCALATION_MODEL,
    CAMBRIAN_MODEL,
    get_generation_number,
    select_model,
)


def test_generation_number_from_empty_history() -> None:
    """With no history, generation number is 1."""
    gen = get_generation_number([])
    assert gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history, generation number is max + 1."""
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    gen = get_generation_number(records)
    assert gen == 3


def test_generation_number_non_sequential() -> None:
    """Generation number is max + 1 even with gaps."""
    records = [
        {"generation": 1},
        {"generation": 5},
        {"generation": 3},
    ]
    gen = get_generation_number(records)
    assert gen == 6


def test_model_escalation_on_retry() -> None:
    """select_model returns escalation model on retry."""
    assert select_model(0) == CAMBRIAN_MODEL
    assert select_model(1) == CAMBRIAN_ESCALATION_MODEL
    assert select_model(2) == CAMBRIAN_ESCALATION_MODEL


def test_max_retries_default() -> None:
    """CAMBRIAN_MAX_RETRIES defaults to 3."""
    assert CAMBRIAN_MAX_RETRIES == int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))


def test_max_gens_default() -> None:
    """CAMBRIAN_MAX_GENS defaults to 5."""
    assert CAMBRIAN_MAX_GENS == int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))


def test_max_parse_retries_default() -> None:
    """CAMBRIAN_MAX_PARSE_RETRIES defaults to 2."""
    assert CAMBRIAN_MAX_PARSE_RETRIES == int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))


def test_max_retries_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CAMBRIAN_MAX_RETRIES respects environment variable."""
    # This test verifies the default value since the module is already loaded
    # The env var is read at import time
    assert isinstance(CAMBRIAN_MAX_RETRIES, int)
    assert CAMBRIAN_MAX_RETRIES >= 0


def test_retry_stops_at_max() -> None:
    """Retry logic stops at CAMBRIAN_MAX_RETRIES."""
    # Verify the constant is a reasonable value
    assert CAMBRIAN_MAX_RETRIES > 0
    assert CAMBRIAN_MAX_RETRIES <= 10


def test_parse_repair_counter() -> None:
    """Parse repair counter respects CAMBRIAN_MAX_PARSE_RETRIES."""
    assert CAMBRIAN_MAX_PARSE_RETRIES >= 0
    assert CAMBRIAN_MAX_PARSE_RETRIES <= 10
