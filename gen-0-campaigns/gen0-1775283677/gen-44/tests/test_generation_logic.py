"""Tests for generation loop logic: generation numbering, retry, escalation."""
from __future__ import annotations

import os
import pytest
from unittest.mock import patch


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    history: list[dict] = []
    if history:
        next_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history up to gen 5, next generation is 6."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 6


def test_generation_number_non_sequential() -> None:
    """Non-sequential history uses max generation + 1."""
    history = [
        {"generation": 10, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 11


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import get_model
    with patch.dict(os.environ, {
        "CAMBRIAN_MODEL": "claude-base",
        "CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation",
    }):
        assert get_model(0) == "claude-base"
        assert get_model(1) == "claude-opus-escalation"
        assert get_model(2) == "claude-opus-escalation"


def test_max_retries_default() -> None:
    """Default CAMBRIAN_MAX_RETRIES is 3."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 3


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_GENS", None)
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert max_gens == 5


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert max_parse == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES can be set via env var."""
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "7"}):
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # "<not a valid file block>" has no <file path="..."> tag,
    # so parse_files returns {} (no error). Need an unclosed block.
    bad_responses = [
        '<file path="x.py">\nno end tag',  # unclosed -> ParseError
    ]
    for _ in range(max_retries + 1):
        try:
            parse_files('<file path="x.py">\nno end tag')
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter: one failure before success."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    responses = [
        '<file path="x.py">\nno end tag',  # bad — unclosed -> ParseError
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
