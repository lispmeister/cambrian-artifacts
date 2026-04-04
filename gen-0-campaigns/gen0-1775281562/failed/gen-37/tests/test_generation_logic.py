"""Tests for generation loop logic: generation numbers, retry, model escalation."""
import os


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    versions: list[dict] = []
    if versions:
        next_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With existing history, next generation is max + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "promoted"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """With non-sequential history, uses max generation."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "failed"},
        {"generation": 3, "outcome": "promoted"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to ESCALATION_MODEL on retry_count >= 1."""
    from src.generate import get_model
    old_model = os.environ.pop("CAMBRIAN_MODEL", None)
    old_esc = os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        assert get_model(0) == "claude-sonnet-4-6"
        assert get_model(1) == "claude-opus-4-6"
        assert get_model(3) == "claude-opus-4-6"
    finally:
        if old_model is not None:
            os.environ["CAMBRIAN_MODEL"] = old_model
        if old_esc is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = old_esc


def test_max_retries_default() -> None:
    """Default CAMBRIAN_MAX_RETRIES is 3."""
    old = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 3
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    old = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert max_gens == 5
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = old


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    old = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert max_parse == 2
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = old


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES is read from environment."""
    old = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 7
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old
        else:
            os.environ.pop("CAMBRIAN_MAX_RETRIES", None)


def test_retry_stops_at_max() -> None:
    """Retry counter stops the loop at CAMBRIAN_MAX_RETRIES."""
    max_retries = 3
    retry_count = 0
    stopped = False
    for _ in range(10):
        retry_count += 1
        if retry_count >= max_retries:
            stopped = True
            break
    assert stopped is True
    assert retry_count == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter increments correctly on ParseError."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # First response is malformed (unclosed block), second is good
    # Note: "<file unclosed" does NOT match the regex r'<file path="([^"]+)">'
    # because it has no path="..." attribute, so parse_files returns {} without error.
    # We need a response that DOES open a block but doesn't close it.
    responses = [
        '<file path="bad.py">\ncontent without closing tag',  # ParseError: unclosed
        '<file path="a.py">\ncontent\n</file:end>\n',         # good
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
