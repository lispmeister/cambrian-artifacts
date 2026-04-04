"""Tests for generation loop logic: generation numbers, retry, model escalation."""
import os


def test_generation_number_from_empty_history() -> None:
    """With no history, first generation is 1."""
    from src.loop import get_next_generation
    assert get_next_generation([]) == 1


def test_generation_number_from_existing_history() -> None:
    """With history, next generation is max+1."""
    from src.loop import get_next_generation
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 2, "outcome": "failed"},
    ]
    assert get_next_generation(records) == 3


def test_generation_number_non_sequential() -> None:
    """Non-sequential history uses max generation."""
    from src.loop import get_next_generation
    records = [
        {"generation": 1, "outcome": "promoted"},
        {"generation": 5, "outcome": "promoted"},
        {"generation": 3, "outcome": "failed"},
    ]
    assert get_next_generation(records) == 6


def test_model_escalation_on_retry() -> None:
    """Model escalates to CAMBRIAN_ESCALATION_MODEL on retry_count >= 1."""
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
    import importlib
    with patch_env({"CAMBRIAN_MAX_RETRIES": "3"}):
        import src.loop as loop_module
        importlib.reload(loop_module)
        assert loop_module.CAMBRIAN_MAX_RETRIES == 3


def test_max_gens_default() -> None:
    """Default CAMBRIAN_MAX_GENS is 5."""
    import importlib
    with patch_env({"CAMBRIAN_MAX_GENS": "5"}):
        import src.loop as loop_module
        importlib.reload(loop_module)
        assert loop_module.CAMBRIAN_MAX_GENS == 5


def test_max_parse_retries_default() -> None:
    """Default CAMBRIAN_MAX_PARSE_RETRIES is 2."""
    import importlib
    with patch_env({"CAMBRIAN_MAX_PARSE_RETRIES": "2"}):
        import src.loop as loop_module
        importlib.reload(loop_module)
        assert loop_module.CAMBRIAN_MAX_PARSE_RETRIES == 2


def test_max_retries_from_env() -> None:
    """CAMBRIAN_MAX_RETRIES can be set from environment."""
    import importlib
    with patch_env({"CAMBRIAN_MAX_RETRIES": "7"}):
        import src.loop as loop_module
        importlib.reload(loop_module)
        assert loop_module.CAMBRIAN_MAX_RETRIES == 7


def test_retry_stops_at_max() -> None:
    """Retry counter stops at CAMBRIAN_MAX_RETRIES."""
    from src.generate import ParseError, parse_files

    max_retries = 3
    failures = 0
    for _ in range(max_retries + 1):
        try:
            parse_files("<not a valid file block>")
        except ParseError:
            failures += 1
            if failures >= max_retries:
                break

    assert failures == max_retries


def test_parse_repair_counter() -> None:
    """Parse repair counter: one failure before success."""
    from src.generate import ParseError, parse_files

    max_parse_retries = 2
    parse_attempts = 0
    parse_success = False

    # First response is malformed (no proper <file path="..."> match and no close)
    # Second response is valid
    responses = [
        '<file path="unclosed.py">\ncontent without close tag',  # bad — no </file:end>
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


from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_env(env_vars: dict) -> any:
    """Context manager to temporarily set environment variables."""
    with patch.dict(os.environ, env_vars):
        yield
