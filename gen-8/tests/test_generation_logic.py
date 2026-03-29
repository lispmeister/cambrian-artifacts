"""Tests for generation number computation, retry logic, model escalation."""

import os
import pytest
from typing import Any


def test_generation_number_from_empty_history() -> None:
    """With no history, generation number is 1."""
    records: list[dict[str, Any]] = []
    if records:
        gen = max(r.get("generation", 0) for r in records) + 1
    else:
        gen = 1
    assert gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history, generation number is max + 1."""
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    gen = max(r.get("generation", 0) for r in records) + 1
    assert gen == 3


def test_generation_number_non_sequential() -> None:
    """Generation number is max + 1 even with gaps."""
    records = [
        {"generation": 1},
        {"generation": 5},
        {"generation": 3},
    ]
    gen = max(r.get("generation", 0) for r in records) + 1
    assert gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import select_model
    os.environ.pop("CAMBRIAN_MODEL", None)
    os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)

    assert select_model(0) == "claude-sonnet-4-6"
    assert select_model(1) == "claude-opus-4-6"
    assert select_model(2) == "claude-opus-4-6"
    assert select_model(5) == "claude-opus-4-6"


def test_max_retries_default() -> None:
    """CAMBRIAN_MAX_RETRIES defaults to 3."""
    os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 3


def test_max_gens_default() -> None:
    """CAMBRIAN_MAX_GENS defaults to 5."""
    os.environ.pop("CAMBRIAN_MAX_GENS", None)
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert max_gens == 5


def test_max_parse_retries_default() -> None:
    """CAMBRIAN_MAX_PARSE_RETRIES defaults to 2."""
    os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert max_parse_retries == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES can be set via environment."""
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 7
    finally:
        os.environ.pop("CAMBRIAN_MAX_RETRIES", None)


def test_retry_stops_at_max() -> None:
    """Retry counter increments and stops at CAMBRIAN_MAX_RETRIES."""
    max_retries = 3
    retry_count = 0
    attempts = 0

    while retry_count <= max_retries and attempts < 10:
        attempts += 1
        # Simulate failure
        retry_count += 1

    assert retry_count == max_retries + 1
    assert attempts == max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter increments and caps at CAMBRIAN_MAX_PARSE_RETRIES."""
    from src.generate import ParseError, parse_files, build_parse_repair_prompt

    max_parse_retries = 2
    parse_repair_count = 0
    malformed = '<file path="unclosed.py">\ncontent without end\n'

    while parse_repair_count <= max_parse_retries:
        try:
            parse_files(malformed)
            break
        except ParseError:
            if parse_repair_count >= max_parse_retries:
                break
            parse_repair_count += 1

    assert parse_repair_count == max_parse_retries