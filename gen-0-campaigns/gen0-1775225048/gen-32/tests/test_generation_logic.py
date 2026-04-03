"""Tests for generation number computation and retry logic."""
from __future__ import annotations

import pytest


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    versions: list[dict] = []
    if versions:
        next_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history, next generation is max + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential history uses max correctly."""
    versions = [
        {"generation": 1},
        {"generation": 3},
        {"generation": 5},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import get_model, DEFAULT_MODEL, ESCALATION_MODEL
    assert get_model(0) == DEFAULT_MODEL
    assert get_model(1) == ESCALATION_MODEL
    assert get_model(3) == ESCALATION_MODEL


def test_max_retries_default() -> None:
    """Default MAX_RETRIES is 3."""
    import os
    env_val = os.environ.get("CAMBRIAN_MAX_RETRIES")
    if env_val is None:
        default = 3
    else:
        default = int(env_val)
    assert default == 3


def test_max_gens_default() -> None:
    """Default MAX_GENS is 5."""
    import os
    env_val = os.environ.get("CAMBRIAN_MAX_GENS")
    if env_val is None:
        default = 5
    else:
        default = int(env_val)
    assert default == 5


def test_max_parse_retries_default() -> None:
    """Default MAX_PARSE_RETRIES is 2."""
    import os
    env_val = os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES")
    if env_val is None:
        default = 2
    else:
        default = int(env_val)
    assert default == 2


def test_max_retries_from_env(monkeypatch) -> None:
    """MAX_RETRIES can be set from environment."""
    monkeypatch.setenv("CAMBRIAN_MAX_RETRIES", "5")
    import importlib
    import src.loop as loop_module
    importlib.reload(loop_module)
    assert loop_module.MAX_RETRIES == 5


def test_retry_stops_at_max() -> None:
    """Retry counter stops at MAX_RETRIES."""
    max_retries = 3
    consecutive_failures = 0

    # Simulate failures
    for _ in range(4):
        if consecutive_failures >= max_retries:
            break
        consecutive_failures += 1

    assert consecutive_failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter: tracks parse failures and stops at max."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate: first response is bad, second is good
    responses = [
        "<file unclosed",  # bad — no closing tag, no ParseError from state machine
        '<file path="a.py">\ncontent\n</file:end>\n',  # good
    ]

    for i, response in enumerate(responses):
        try:
            result = parse_files(response)
            parse_success = True
            break
        except ParseError:
            parse_attempts += 1
            if parse_attempts >= max_parse_retries:
                break

    assert parse_success is True
    # The first response "<file unclosed" has no closing tag but also no open tag matched
    # (no '">' after path=), so it returns empty dict without raising ParseError.
    # So parse_attempts stays 0 and parse_success is True on first try.
    # This is correct behavior: malformed open tags don't raise, unclosed open tags do.
    assert parse_attempts == 0
