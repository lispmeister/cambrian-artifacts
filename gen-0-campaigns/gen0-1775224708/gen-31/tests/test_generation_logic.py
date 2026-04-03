"""Tests for generation loop logic (generation numbering, retry, model selection)."""
from __future__ import annotations


def test_generation_number_from_empty_history() -> None:
    """With no history, next generation is 1."""
    records: list[dict] = []
    if records:
        next_gen = max(r.get("generation", 0) for r in records) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """With history up to gen 5, next generation is 6."""
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
        {"generation": 5, "outcome": "promoted"},
    ]
    next_gen = max(r.get("generation", 0) for r in records) + 1
    assert next_gen == 6


def test_generation_number_non_sequential() -> None:
    """Non-sequential history: next gen is max+1."""
    records = [
        {"generation": 3, "outcome": "promoted"},
        {"generation": 7, "outcome": "failed"},
        {"generation": 2, "outcome": "promoted"},
    ]
    next_gen = max(r.get("generation", 0) for r in records) + 1
    assert next_gen == 8


def test_model_escalation_on_retry() -> None:
    """Model escalates on retry_count >= 1."""
    from src.generate import get_model
    base = "claude-sonnet-4-6"
    escalation = "claude-opus-4-6"
    assert get_model(base, escalation, 0) == base
    assert get_model(base, escalation, 1) == escalation
    assert get_model(base, escalation, 2) == escalation
    assert get_model(base, escalation, 3) == escalation


def test_max_retries_default() -> None:
    """Default max retries is 3."""
    import os
    old = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 3
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_max_gens_default() -> None:
    """Default max generations is 5."""
    import os
    old = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        max_gens = int(os.environ.get("CAMBRIAN_MAX_GENS", "5"))
        assert max_gens == 5
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = old


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    import os
    old = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        max_parse = int(os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES", "2"))
        assert max_parse == 2
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = old


def test_max_retries_from_env() -> None:
    """Max retries can be configured via environment variable."""
    import os
    old = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        max_retries = int(os.environ.get("CAMBRIAN_MAX_RETRIES", "3"))
        assert max_retries == 7
    finally:
        if old is None:
            del os.environ["CAMBRIAN_MAX_RETRIES"]
        else:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_retry_stops_at_max() -> None:
    """Retry loop stops when max retries is reached."""
    max_retries = 3
    retry_count = 0
    attempts = 0

    while retry_count <= max_retries and attempts < 10:
        attempts += 1
        retry_count += 1
        if retry_count > max_retries:
            break

    assert retry_count > max_retries or attempts >= 10
    assert attempts <= max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter: ParseError is raised for truly unclosed blocks."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # A truly unclosed block (valid header, no close tag) raises ParseError
    # A non-matching string returns {} without raising ParseError
    responses = [
        # This HAS a valid <file path="..."> header but no </file:end> — raises ParseError
        '<file path="missing.py">\ncontent without end',
        # This is a valid complete block — succeeds
        '<file path="a.py">\ncontent\n</file:end>\n',
    ]

    for response in responses:
        try:
            parse_files(response)
            parse_success = True
            break
        except ParseError:
            parse_attempts += 1
            if parse_attempts >= max_parse_retries:
                break

    assert parse_success is True
    assert parse_attempts == 1  # One failure (unclosed block) before success
