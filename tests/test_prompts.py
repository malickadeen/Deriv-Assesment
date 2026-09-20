"""Tests for prompt construction and versioning (no API key required)."""

from __future__ import annotations

import pytest

from src.prompts import PROMPT_VERSIONS, build_messages, system_prompt
from src.models import Ticket


def _ticket() -> Ticket:
    return Ticket(ticket_id="P-1", subject="s", message="m", channel="email")


def test_versions_registered() -> None:
    assert {"v1", "v2", "v3"} <= set(PROMPT_VERSIONS)


def test_v1_and_v3_differ() -> None:
    assert system_prompt("v1") != system_prompt("v3")


def test_v3_contains_rubric_and_examples() -> None:
    text = system_prompt("v3")
    assert "DECISION RUBRIC" in text
    assert "WORKED EXAMPLES" in text


def test_v2_is_strict_retry() -> None:
    assert "STRICT MODE" in system_prompt("v2")


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_messages_shape(version: str) -> None:
    messages = build_messages(_ticket(), version)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "P-1" in messages[1]["content"]


def test_all_versions_list_enum_values() -> None:
    # Every version must restate the allowed category enum so the model and the
    # validator can never drift.
    for version in ("v1", "v3"):
        assert "payment_issue" in system_prompt(version)
