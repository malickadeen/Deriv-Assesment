"""Tests for deterministic prediction validation.

No API key is required. Temporary directories are used for artifacts so tests
never depend on a real run or on specific sample ticket text.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models import Ticket
from src.validator import validate_prediction_file, validate_predictions


def _ticket(tid: str) -> Ticket:
    return Ticket(ticket_id=tid, subject="s", message="m", channel="email")


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_valid_prediction_passes(tmp_path: Path) -> None:
    p = tmp_path / "X-1.json"
    _write(p, {
        "ticket_id": "X-1", "category": "payment_issue",
        "priority": "high", "sentiment": "frustrated",
        "reasoning": "Duplicate charge reported.",
    })
    result = validate_prediction_file(p, "X-1")
    assert result.valid, result.errors
    assert result.errors == []


def test_invalid_enum_fails(tmp_path: Path) -> None:
    p = tmp_path / "X-2.json"
    _write(p, {
        "ticket_id": "X-2", "category": "payment_issue",
        "priority": "critical",  # not an allowed enum value
        "sentiment": "frustrated", "reasoning": "x",
    })
    result = validate_prediction_file(p, "X-2")
    assert not result.valid
    assert any("Invalid priority" in e for e in result.errors)


def test_ticket_id_mismatch_detected(tmp_path: Path) -> None:
    p = tmp_path / "X-3.json"
    _write(p, {
        "ticket_id": "WRONG", "category": "other",
        "priority": "low", "sentiment": "neutral", "reasoning": "x",
    })
    result = validate_prediction_file(p, "X-3")
    assert not result.valid
    assert any("mismatch" in e for e in result.errors)


def test_missing_field_detected(tmp_path: Path) -> None:
    p = tmp_path / "X-4.json"
    _write(p, {  # no "reasoning"
        "ticket_id": "X-4", "category": "other",
        "priority": "low", "sentiment": "neutral",
    })
    result = validate_prediction_file(p, "X-4")
    assert not result.valid
    assert any("reasoning" in e for e in result.errors)


def test_missing_file_detected(tmp_path: Path) -> None:
    result = validate_prediction_file(tmp_path / "nope.json", "X-5")
    assert not result.valid
    assert any("does not exist" in e for e in result.errors)


def test_failure_artifact_is_invalid(tmp_path: Path) -> None:
    p = tmp_path / "X-6.json"
    _write(p, {"ticket_id": "X-6", "error": "invalid_json", "valid": False})
    result = validate_prediction_file(p, "X-6")
    assert not result.valid


def test_validate_predictions_aggregate(tmp_path: Path) -> None:
    _write(tmp_path / "A.json", {
        "ticket_id": "A", "category": "other", "priority": "low",
        "sentiment": "neutral", "reasoning": "ok",
    })
    _write(tmp_path / "B.json", {
        "ticket_id": "B", "category": "other", "priority": "nope",
        "sentiment": "neutral", "reasoning": "bad",
    })
    report = validate_predictions([_ticket("A"), _ticket("B")], tmp_path)
    assert report.total_tickets == 2
    assert report.valid_outputs == 1
    assert report.invalid_outputs == 1
