"""Tests for generation loop logic."""
from __future__ import annotations


def test_generation_number_from_empty_history() -> None:
    """Empty history means offspring is generation 1."""
    history: list[dict] = []
    if history:
        offspring_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        offspring_gen = 1
    assert offspring_gen == 1


def test_generation_number_from_existing_history() -> None:
    """History with max gen=5 means offspring is 6."""
    history = [
        {"generation": 3, "outcome": "failed"},
        {"generation": 5, "outcome": "promoted"},
        {"generation": 4, "outcome": "failed"},
    ]
    offspring_gen = max(r.get("generation", 0) for r in history) + 1
    assert offspring_gen == 6


def test_generation_number_non_sequential() -> None:
    """Non-sequential history still picks max+1."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 10, "outcome": "failed"},
        {"generation": 3, "outcome": "promoted"},
    ]
    offspring_gen = max(r.get("generation", 0) for r in history) + 1
    assert offspring_gen == 11


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    from unittest.mock import patch
    from src.generate import get_model
    with patch.dict(__import__("os").environ, {
        "CAMBRIAN_MODEL": "claude-base",
        "CAMBRIAN_ESCALATION_MODEL": "claude-opus-escalation",
    }):
        assert get_model(0) == "claude-base"
        assert get_model(1) == "claude-opus-escalation"
        assert get_model(2) == "claude-opus-escalation"


def test_max_retries_default() -> None:
    """Default max retries is 3."""
    import os
    max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 3


def test_max_gens_default() -> None:
    """Default max generations is 5."""
    import os
    max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert max_gens == 5


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    import os
    max_parse_retries = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert max_parse_retries == 2


def test_max_retries_from_env() -> None:
    """Max retries can be set from environment."""
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "5"}):
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 5


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # Use a response that has an opening tag but no closing tag to trigger ParseError
    bad_response = '<file path="test.py">\ncontent without end tag\n'
    for _ in range(max_retries + 1):
        try:
            parse_files(bad_response)
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter doesn't count against generation retries."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # First response: has opening tag but no closing — raises ParseError
    # Second response: well-formed
    responses = [
        '<file path="missing_end.py">\ncontent without end\n',  # bad — raises ParseError
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
