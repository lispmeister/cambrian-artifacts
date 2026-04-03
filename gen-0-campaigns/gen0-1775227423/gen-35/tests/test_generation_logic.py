"""Tests for generation loop logic."""
from __future__ import annotations

import os


def test_generation_number_from_empty_history() -> None:
    """Empty history produces generation 1."""
    from src.generate import compute_next_generation
    assert compute_next_generation([]) == 1


def test_generation_number_from_existing_history() -> None:
    """Next generation is max + 1."""
    from src.generate import compute_next_generation
    records = [{"generation": 1}, {"generation": 2}]
    assert compute_next_generation(records) == 3


def test_generation_number_non_sequential() -> None:
    """Works with gaps in generation numbers."""
    from src.generate import compute_next_generation
    records = [{"generation": 1}, {"generation": 5}]
    assert compute_next_generation(records) == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates on retry."""
    from src.generate import select_model
    old_model = os.environ.get("CAMBRIAN_MODEL")
    old_esc = os.environ.get("CAMBRIAN_ESCALATION_MODEL")
    os.environ.pop("CAMBRIAN_MODEL", None)
    os.environ.pop("CAMBRIAN_ESCALATION_MODEL", None)
    try:
        assert select_model(0) == "claude-sonnet-4-6"
        assert select_model(1) == "claude-opus-4-6"
        assert select_model(3) == "claude-opus-4-6"
    finally:
        if old_model is not None:
            os.environ["CAMBRIAN_MODEL"] = old_model
        if old_esc is not None:
            os.environ["CAMBRIAN_ESCALATION_MODEL"] = old_esc


def test_max_retries_default() -> None:
    """Default max retries is 3."""
    from src.generate import get_max_retries
    old = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        assert get_max_retries() == 3
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_max_gens_default() -> None:
    """Default max gens is 5."""
    from src.generate import get_max_gens
    old = os.environ.get("CAMBRIAN_MAX_GENS")
    os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        assert get_max_gens() == 5
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = old


def test_max_parse_retries_default() -> None:
    """Default max parse retries is 2."""
    from src.generate import get_max_parse_retries
    old = os.environ.get("CAMBRIAN_MAX_PARSE_RETRIES")
    os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        assert get_max_parse_retries() == 2
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = old


def test_max_retries_from_env() -> None:
    """Max retries reads from env."""
    from src.generate import get_max_retries
    old = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        assert get_max_retries() == 7
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old
        else:
            os.environ.pop("CAMBRIAN_MAX_RETRIES", None)


def test_retry_stops_at_max() -> None:
    """Retry counter respects max retries limit."""
    from src.generate import get_max_retries
    old = os.environ.get("CAMBRIAN_MAX_RETRIES")
    os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        max_retries = get_max_retries()
        assert max_retries == 3
        # Simulate: after 3 consecutive failures, we should stop
        consecutive_failures = 0
        for _ in range(5):
            consecutive_failures += 1
            if consecutive_failures >= max_retries:
                break
        assert consecutive_failures == 3
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_parse_repair_counter() -> None:
    """Parse repair counter doesn't count against generation retries."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # Simulate parse failure then success.
    # The first response opens a <file> block without closing it -> ParseError.
    # The second response is valid.
    responses = [
        '<file path="a.py">\ncontent without closing tag\n',
        '<file path="a.py">\ncontent\n</file:end>\n',
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
    assert parse_attempts == 1
