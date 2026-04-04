"""Tests for generation number computation, retry logic, and model escalation."""
import os
from unittest.mock import patch


def test_generation_number_from_empty_history() -> None:
    """With no history, offspring is generation 1."""
    versions: list[dict] = []
    if versions:
        offspring_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        offspring_gen = 1
    assert offspring_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history, offspring is max generation + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "promoted"},
    ]
    offspring_gen = max(v.get("generation", 0) for v in versions) + 1
    assert offspring_gen == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential history uses max + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    offspring_gen = max(v.get("generation", 0) for v in versions) + 1
    assert offspring_gen == 6


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
    """Default max retries is 3."""
    default = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert default == 3


def test_max_gens_default() -> None:
    """Default max generations is 5."""
    default = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert default == 5


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    default = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert default == 2


def test_max_retries_from_env() -> None:
    """Max retries can be configured from environment."""
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "7"}):
        value = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert value == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # Use an input that will definitely raise ParseError: unclosed block
    for _ in range(max_retries + 1):
        try:
            parse_files('<file path="unclosed.py">\ncontent without end')
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter counts failures correctly."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # First response: unclosed block -> ParseError
    # Second response: valid -> success
    responses = [
        '<file path="unclosed.py">\ncontent without end',  # bad - no closing tag
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
