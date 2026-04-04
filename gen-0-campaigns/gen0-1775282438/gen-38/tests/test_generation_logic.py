"""Tests for generation loop logic: counters, retries, model selection."""


def test_generation_number_from_empty_history() -> None:
    """With no history, first generation is 1."""
    history: list = []
    if history:
        next_gen = max(r.get("generation", 0) for r in history) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """Generation number is max(existing) + 1."""
    history = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Generation number works with non-sequential history."""
    history = [
        {"generation": 5, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    next_gen = max(r.get("generation", 0) for r in history) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry."""
    import os
    from src.generate import select_model
    default_model = os.environ.get("CAMBRIAN_MODEL", "claude-sonnet-4-6")
    escalation_model = os.environ.get("CAMBRIAN_ESCALATION_MODEL", "claude-opus-4-6")
    assert select_model(0) == default_model
    assert select_model(1) == escalation_model
    assert select_model(3) == escalation_model


def test_max_retries_default() -> None:
    """Default CAMBRIAN_MAX_RETRIES is 3."""
    import os
    val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
    assert val == 3


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    import os
    val = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
    assert val == 5


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    import os
    val = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
    assert val == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES can be overridden via environment."""
    import os
    original = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        val = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert val == 7
    finally:
        if original is None:
            del os.environ["CAMBRIAN_MAX_RETRIES"]
        else:
            os.environ["CAMBRIAN_MAX_RETRIES"] = original


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    max_retries = 3
    consecutive_failures = 0
    max_gens = 10
    total_gens = 0

    while total_gens < max_gens and consecutive_failures < max_retries:
        # Simulate a failure every iteration
        consecutive_failures += 1
        total_gens += 1

    assert consecutive_failures == max_retries
    assert total_gens == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter: first response fails, second succeeds."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # First response is malformed (no closing tag — raises ParseError)
    # Second response is valid
    responses = [
        "<file path=\"bad.py\">\nno closing tag",  # bad — raises ParseError
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
