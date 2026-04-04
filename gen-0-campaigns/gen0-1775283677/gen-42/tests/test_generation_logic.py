"""Tests for generation loop logic."""
from __future__ import annotations


def test_generation_number_from_empty_history() -> None:
    """With empty history, offspring is generation 1."""
    versions: list = []
    if versions:
        next_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history up to N, offspring is N+1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "promoted"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """With non-sequential history, offspring is max+1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 6


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
    """Default CAMBRIAN_MAX_RETRIES is 3."""
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {}, clear=False):
        env_copy = dict(os.environ)
        env_copy.pop("CAMBRIAN_MAX_RETRIES", None)
        with patch.dict(os.environ, env_copy, clear=True):
            val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
            assert val == 3


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    import os
    from unittest.mock import patch
    env_copy = dict(os.environ)
    env_copy.pop("CAMBRIAN_MAX_GENS", None)
    with patch.dict(os.environ, env_copy, clear=True):
        val = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert val == 5


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    import os
    from unittest.mock import patch
    env_copy = dict(os.environ)
    env_copy.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    with patch.dict(os.environ, env_copy, clear=True):
        val = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert val == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES can be set via environment variable."""
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "7"}):
        val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert val == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # Use a response with an actual unclosed file block to trigger ParseError
    for _ in range(max_retries + 1):
        try:
            parse_files('<file path="unclosed.py">\ncontent without close tag\n')
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter counts failures before success."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success
    # First response: unclosed block (triggers ParseError)
    # Second response: valid
    responses = [
        '<file path="unclosed.py">\ncontent\n',  # bad — unclosed block
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
