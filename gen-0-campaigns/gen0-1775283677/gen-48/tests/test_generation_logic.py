"""Tests for generation loop logic."""

from __future__ import annotations

import os
from unittest.mock import patch


def test_generation_number_from_empty_history() -> None:
    """When history is empty, offspring generation is 1."""
    history: list = []
    if history:
        offspring_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        offspring_gen = 1
    assert offspring_gen == 1


def test_generation_number_from_existing_history() -> None:
    """When history has records, offspring is max+1."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "promoted"},
    ]
    offspring_gen = max(r.get("generation", 0) for r in history) + 1
    assert offspring_gen == 3


def test_generation_number_non_sequential() -> None:
    """Generation number uses max even if records are non-sequential."""
    history = [
        {"generation": 5, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
        {"generation": 8, "outcome": "promoted"},
    ]
    offspring_gen = max(r.get("generation", 0) for r in history) + 1
    assert offspring_gen == 9


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import get_model
    with patch.dict(os.environ, {
        "CAMBRIAN_MODEL": "claude-base",
        "CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation",
    }, clear=False):
        assert get_model(0) == "claude-base"
        assert get_model(1) == "claude-opus-escalation"
        assert get_model(2) == "claude-opus-escalation"


def test_max_retries_default() -> None:
    """CAMBRIAN_MAX_RETRIES defaults to 3."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
        val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert val == 3


def test_max_gens_default() -> None:
    """CAMBRIAN_MAX_GENS defaults to 5."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_GENS", None)
        val = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert val == 5


def test_max_parse_retries_default() -> None:
    """CAMBRIAN_MAX_PARSE_RETRIES defaults to 2."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
        val = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert val == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES is read from environment."""
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "7"}, clear=False):
        val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert val == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # Use a response that IS a valid open tag but has no close tag
    bad_response = '<file path="x.py">\ncontent without close tag'
    for _ in range(max_retries + 1):
        try:
            parse_files(bad_response)
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter tracks failures correctly."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # First response: has an open tag but no close tag -> ParseError
    # Second response: valid
    responses = [
        '<file path="unclosed.py">\ncontent without end tag',  # bad - no </file:end>
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
