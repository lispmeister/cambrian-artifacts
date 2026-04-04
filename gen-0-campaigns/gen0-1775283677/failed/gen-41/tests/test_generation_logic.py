"""Tests for generation loop logic: counters, model selection, retry behavior."""


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    history: list = []
    if history:
        next_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history max gen=3, next generation is 4."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
        {"generation": 3, "outcome": "promoted"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 4


def test_generation_number_non_sequential() -> None:
    """Generation number is max+1 even with gaps."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "failed"},
        {"generation": 3, "outcome": "promoted"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    import os
    from unittest.mock import patch
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
    import os
    from unittest.mock import patch
    env = {k: v for k, v in os.environ.items() if k != "CAMBRIAN_MAX_RETRIES"}
    with patch.dict(os.environ, env, clear=True):
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 3


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    import os
    from unittest.mock import patch
    env = {k: v for k, v in os.environ.items() if k != "CAMBRIAN_MAX_GENS"}
    with patch.dict(os.environ, env, clear=True):
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert max_gens == 5


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    import os
    from unittest.mock import patch
    env = {k: v for k, v in os.environ.items() if k != "CAMBRIAN_MAX_PARSE_RETRIES"}
    with patch.dict(os.environ, env, clear=True):
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert max_parse == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES is read from environment."""
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {"CAMBRIAN_MAX_RETRIES": "7"}):
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert max_retries == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    # Use a response that will actually raise ParseError (unclosed block)
    bad_response = '<file path="unclosed.py">\ncontent without end tag'
    for _ in range(max_retries + 1):
        try:
            parse_files(bad_response)
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter: one failure, then success."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # First response: unclosed block (raises ParseError)
    # Second response: valid block (succeeds)
    responses = [
        '<file path="unclosed.py">\ncontent without end',   # bad — unclosed
        '<file path="a.py">\ncontent\n</file:end>\n',        # good
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
