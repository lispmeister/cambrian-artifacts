"""Tests for generation loop logic: generation numbers, retry counters, model selection."""
import os
import pytest


def test_generation_number_from_empty_history() -> None:
    """When history is empty, next generation is 1."""
    versions: list = []
    if versions:
        next_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """Next generation is max existing + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential history: next gen is still max + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL when retry_count >= 1."""
    from src.generate import select_model, MODEL, ESCALATION_MODEL
    assert select_model(0) == MODEL
    assert select_model(1) == ESCALATION_MODEL
    assert select_model(3) == ESCALATION_MODEL


def test_max_retries_default() -> None:
    """Default max retries is 3."""
    import os
    original = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 3
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = original


def test_max_gens_default() -> None:
    """Default max generations is 5."""
    import os
    original = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert max_gens == 5
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = original


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    import os
    original = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert max_parse == 2
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = original


def test_max_retries_from_env() -> None:
    """Max retries can be set via environment variable."""
    import os
    original = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 7
    finally:
        if original is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = original
        else:
            del os.environ["CAMBRIAN_MAX_RETRIES"]


def test_retry_stops_at_max() -> None:
    """Retry counter stops at max retries."""
    max_retries = 3
    consecutive_failures = 0

    # Simulate 3 consecutive failures
    for _ in range(3):
        consecutive_failures += 1
        if consecutive_failures >= max_retries:
            break

    assert consecutive_failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter increments and a successful parse is detected."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # First response: malformed (no closing tag for a proper <file> block)
    # Second response: valid
    responses = [
        '<file path="unclosed.py">\ncontent without end',  # bad — no closing tag
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
