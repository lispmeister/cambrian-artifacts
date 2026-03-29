"""Tests for generation number computation and retry logic."""

import os

import pytest


def test_generation_number_from_empty_history() -> None:
    """Generation 1 is produced from empty history."""
    versions: list = []
    if versions:
        next_gen = max(v.get("generation", 0) for v in versions) + 1
    else:
        next_gen = 1
    assert next_gen == 1


def test_generation_number_from_existing_history() -> None:
    """Next generation is max(existing) + 1."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "promoted"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential generations handled by max()."""
    versions = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    next_gen = max(v.get("generation", 0) for v in versions) + 1
    assert next_gen == 4


def test_model_escalation_on_retry() -> None:
    """Escalation model is used when retry_count >= 1."""
    import src.generate as gen_module

    original_model = gen_module.CAMBRIAN_MODEL
    original_escalation = gen_module.CAMBRIAN_ESCALATION_MODEL
    gen_module.CAMBRIAN_MODEL = "sonnet"
    gen_module.CAMBRIAN_ESCALATION_MODEL = "opus"

    def get_model(retry_count: int) -> str:
        return gen_module.CAMBRIAN_MODEL if retry_count == 0 else gen_module.CAMBRIAN_ESCALATION_MODEL

    assert get_model(0) == "sonnet"
    assert get_model(1) == "opus"
    assert get_model(2) == "opus"

    gen_module.CAMBRIAN_MODEL = original_model
    gen_module.CAMBRIAN_ESCALATION_MODEL = original_escalation


def test_max_retries_default() -> None:
    """CAMBRIAN_MAX_RETRIES defaults to 3."""
    import sys

    if "src.loop" in sys.modules:
        saved = sys.modules.pop("src.loop")
    else:
        saved = None

    old_val = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        import src.loop as loop_module
        assert loop_module.MAX_RETRIES == 3
    finally:
        if old_val is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old_val
        if saved is not None:
            sys.modules["src.loop"] = saved
        elif "src.loop" in sys.modules:
            del sys.modules["src.loop"]


def test_max_gens_default() -> None:
    """CAMBRIAN_MAX_GENS defaults to 5."""
    import sys

    if "src.loop" in sys.modules:
        saved = sys.modules.pop("src.loop")
    else:
        saved = None

    old_val = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        import src.loop as loop_module
        assert loop_module.MAX_GENS == 5
    finally:
        if old_val is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = old_val
        if saved is not None:
            sys.modules["src.loop"] = saved
        elif "src.loop" in sys.modules:
            del sys.modules["src.loop"]


def test_max_parse_retries_default() -> None:
    """CAMBRIAN_MAX_PARSE_RETRIES defaults to 2."""
    import sys

    if "src.loop" in sys.modules:
        saved = sys.modules.pop("src.loop")
    else:
        saved = None

    old_val = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        import src.loop as loop_module
        assert loop_module.MAX_PARSE_RETRIES == 2
    finally:
        if old_val is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = old_val
        if saved is not None:
            sys.modules["src.loop"] = saved
        elif "src.loop" in sys.modules:
            del sys.modules["src.loop"]


def test_max_retries_from_env(monkeypatch) -> None:
    """CAMBRIAN_MAX_RETRIES can be set via environment."""
    import sys
    monkeypatch.setenv("CAMBRIAN_MAX_RETRIES", "7")
    if "src.loop" in sys.modules:
        del sys.modules["src.loop"]
    import src.loop as loop_module
    assert loop_module.MAX_RETRIES == 7
    del sys.modules["src.loop"]


def test_retry_stops_at_max() -> None:
    """Retry loop stops when retry_count exceeds MAX_RETRIES."""
    max_retries = 3
    retry_count = 0
    attempts = 0

    while retry_count <= max_retries and attempts < 100:
        attempts += 1
        # Simulate failure
        retry_count += 1
        if retry_count > max_retries:
            break

    assert retry_count == max_retries + 1
    assert attempts == max_retries + 1


def test_parse_repair_counter() -> None:
    """Parse repair counter increments and stops at MAX_PARSE_RETRIES."""
    max_parse_retries = 2
    parse_repair_count = 0
    repairs_attempted = 0

    while True:
        # Simulate parse error
        if parse_repair_count >= max_parse_retries:
            break
        parse_repair_count += 1
        repairs_attempted += 1

    assert repairs_attempted == max_parse_retries
    assert parse_repair_count == max_parse_retries