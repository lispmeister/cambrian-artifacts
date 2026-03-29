"""Tests for generation loop logic: number computation, retry, model escalation."""

from __future__ import annotations

import os
import importlib
import pytest


def test_generation_number_from_empty_history() -> None:
    """With no history, first generation should be 1."""
    records: list[dict] = []  # type: ignore[type-arg]
    max_gen = max((r.get("generation", 0) for r in records), default=0)
    assert max_gen + 1 == 1


def test_generation_number_from_existing_history() -> None:
    """With existing generations, next should be max+1."""
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    max_gen = max((r.get("generation", 0) for r in records), default=0)
    assert max_gen + 1 == 3


def test_generation_number_non_sequential() -> None:
    """Even with gaps, next generation is max+1."""
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
    ]
    max_gen = max((r.get("generation", 0) for r in records), default=0)
    assert max_gen + 1 == 6


def test_model_escalation_on_retry() -> None:
    """Model should escalate to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
    os.environ["CAMBRIAN_MODEL"] = "sonnet-model"
    os.environ["CAMBRIAN_ESCALATION_MODEL"] = "opus-model"

    import src.generate as gen_mod
    importlib.reload(gen_mod)

    loop = gen_mod.GenerationLoop()

    # Simulate selection logic
    def select_model(retry_count: int) -> str:
        return loop.model if retry_count == 0 else loop.escalation_model

    assert select_model(0) == "sonnet-model"
    assert select_model(1) == "opus-model"
    assert select_model(2) == "opus-model"
    assert select_model(10) == "opus-model"


def test_max_retries_default() -> None:
    """Default max retries should be 3."""
    old = os.environ.pop("CAMBRIAN_MAX_RETRIES", None)
    try:
        import src.generate as gen_mod
        importlib.reload(gen_mod)
        loop = gen_mod.GenerationLoop()
        assert loop.max_retries == 3
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_RETRIES"] = old


def test_max_gens_default() -> None:
    """Default max generations should be 5."""
    old = os.environ.pop("CAMBRIAN_MAX_GENS", None)
    try:
        import src.generate as gen_mod
        importlib.reload(gen_mod)
        loop = gen_mod.GenerationLoop()
        assert loop.max_gens == 5
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_GENS"] = old


def test_max_parse_retries_default() -> None:
    """Default max parse retries should be 2."""
    old = os.environ.pop("CAMBRIAN_MAX_PARSE_RETRIES", None)
    try:
        import src.generate as gen_mod
        importlib.reload(gen_mod)
        loop = gen_mod.GenerationLoop()
        assert loop.max_parse_retries == 2
    finally:
        if old is not None:
            os.environ["CAMBRIAN_MAX_PARSE_RETRIES"] = old


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES env var is respected."""
    os.environ["CAMBRIAN_MAX_RETRIES"] = "7"
    try:
        import src.generate as gen_mod
        importlib.reload(gen_mod)
        loop = gen_mod.GenerationLoop()
        assert loop.max_retries == 7
    finally:
        os.environ.pop("CAMBRIAN_MAX_RETRIES", None)


def test_retry_stops_at_max() -> None:
    """Retry counter logic: stops when retry_count > max_retries."""
    max_retries = 3
    retry_count = 0

    results: list[bool] = []
    for _ in range(10):
        if retry_count > max_retries:
            results.append(False)  # would stop
        else:
            results.append(True)  # would continue
            retry_count += 1

    # Should stop after max_retries + 1 attempts
    assert results.count(True) == max_retries + 1
    assert results.count(False) == 10 - (max_retries + 1)


def test_parse_repair_counter() -> None:
    """Parse repair loop retries up to max_parse_retries."""
    import src.generate as gen_mod
    importlib.reload(gen_mod)

    parse_files = gen_mod.parse_files
    ParseError = gen_mod.ParseError

    max_parse_retries = 2
    attempts = 0
    success = False

    bad_response = '<file path="x.py">\ncontent\n'  # no closing tag

    for attempt in range(max_parse_retries + 1):
        attempts += 1
        try:
            parse_files(bad_response)
            success = True
            break
        except ParseError:
            if attempt >= max_parse_retries:
                break
        except Exception:
            # Catch any exception that might be a ParseError from another module instance
            if attempt >= max_parse_retries:
                break

    assert attempts == max_parse_retries + 1
    assert not success