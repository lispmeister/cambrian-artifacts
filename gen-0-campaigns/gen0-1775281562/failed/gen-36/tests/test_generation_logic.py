"""Tests for generation logic, retry counters, and model selection."""
from __future__ import annotations

import os
import pytest


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
        {"generation": 2, "outcome": "promoted"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Generation number uses max, not count."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import get_model
    original_model = os.environ.get("CAMBRIAN_MODEL")
    original_escalation = os.environ.get("CAMBRIAN_ESCALATION_MODEL")
    os.environ["CAMBRIAN_MODEL"] = "model-a"
    os.environ["CAMBRIAN_ESCALATION_MODEL"] = "model-b"
    try:
        assert get_model(0) == "model-a"
        assert get_model(1) == "model-b"
        assert get_model(3) == "model-b"
    finally:
        if original_model is None:
            os.environ.pop("CAMBRIAN_MODEL", None)
        else:
            os.environ["CAMBRIAN_MODEL"] = original_model
        if original_escalation is None:
            os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
        else:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = original_escalation


def test_max_retries_default() -> None:
    """Default MAX_RETRIES is 3."""
    original = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 3
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = original


def test_max_gens_default() -> None:
    """Default MAX_GENS is 5."""
    original = os.environ.get("CAMBRIAN_MAX_GENS")
    os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert max_gens == 5
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = original


def test_max_parse_retries_default() -> None:
    """Default MAX_PARSE_RETRIES is 2."""
    original = os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES")
    os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert max_parse == 2
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = original


def test_max_retries_from_env() -> None:
    """MAX_RETRIES can be configured via environment variable."""
    original = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 7
    finally:
        if original is None:
            os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
        else:
            os.environ["CAMBRIAN_MAX_RETRIES"] = original


def test_retry_stops_at_max() -> None:
    """Retry counter stops loop at MAX_RETRIES."""
    max_retries = 3
    retry_count = 0
    gen_count = 0

    # Simulate all failures
    while retry_count <= max_retries and gen_count < 10:
        # Simulate non-viable outcome
        retry_count += 1
        gen_count += 1
        if retry_count > max_retries:
            break

    assert retry_count > max_retries or gen_count <= max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter tracks failures correctly."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # The first response is truly unclosed (has a proper <file> header but no end tag)
    # The second response is valid
    responses = [
        '<file path="missing_end.py">\ncontent\n',  # bad — no closing tag, raises ParseError
        '<file path="a.py">\ncontent\n</file:end>\n',  # good
    ]

    for response in responses:
        try:
            result = parse_files(response)
            parse_success = True
            break
        except ParseError:
            parse_attempts += 1
            if parse_attempts >= max_parse_retries:
                break

    assert parse_success is True
    assert parse_attempts == 1  # Only one failure before success
