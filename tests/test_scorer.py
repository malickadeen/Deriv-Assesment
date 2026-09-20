"""Tests for deterministic scoring, including the missing-label and
invalid-prediction policies. No API key required."""

from __future__ import annotations

import json
from pathlib import Path

from src.loader import prediction_path
from src.models import Label, Ticket
from src.scorer import score
from src.validator import validate_predictions


def _ticket(tid: str) -> Ticket:
    return Ticket(ticket_id=tid, subject="s", message="m", channel="email")


def _write_pred(output_dir: Path, tid: str, **fields) -> None:
    obj = {"ticket_id": tid, "reasoning": "r", **fields}
    prediction_path(output_dir, tid).write_text(json.dumps(obj), encoding="utf-8")


def _label(cat: str, pri: str, sen: str) -> Label:
    return Label(category=cat, priority=pri, sentiment=sen)


def test_all_correct_perfect_scores(tmp_path: Path) -> None:
    tickets = [_ticket("A"), _ticket("B")]
    _write_pred(tmp_path, "A", category="payment_issue", priority="high", sentiment="frustrated")
    _write_pred(tmp_path, "B", category="other", priority="low", sentiment="neutral")
    labels = {
        "A": _label("payment_issue", "high", "frustrated"),
        "B": _label("other", "low", "neutral"),
    }
    validation = validate_predictions(tickets, tmp_path)
    metrics = score(tickets, labels, tmp_path, validation)
    assert metrics.scorable_tickets == 2
    assert metrics.category_accuracy == 1.0
    assert metrics.exact_match_rate == 1.0


def test_partial_accuracy(tmp_path: Path) -> None:
    tickets = [_ticket("A"), _ticket("B"), _ticket("C")]
    # A fully correct; B wrong priority; C wrong sentiment.
    _write_pred(tmp_path, "A", category="other", priority="low", sentiment="neutral")
    _write_pred(tmp_path, "B", category="other", priority="high", sentiment="neutral")
    _write_pred(tmp_path, "C", category="other", priority="low", sentiment="urgent")
    labels = {
        "A": _label("other", "low", "neutral"),
        "B": _label("other", "low", "neutral"),
        "C": _label("other", "low", "neutral"),
    }
    validation = validate_predictions(tickets, tmp_path)
    metrics = score(tickets, labels, tmp_path, validation)
    assert metrics.scorable_tickets == 3
    assert metrics.category_accuracy == 1.0
    assert metrics.priority_accuracy == round(2 / 3, 4)
    assert metrics.sentiment_accuracy == round(2 / 3, 4)
    assert metrics.exact_match_rate == round(1 / 3, 4)


def test_missing_label_not_scorable(tmp_path: Path) -> None:
    tickets = [_ticket("A"), _ticket("B")]
    _write_pred(tmp_path, "A", category="other", priority="low", sentiment="neutral")
    _write_pred(tmp_path, "B", category="other", priority="low", sentiment="neutral")
    labels = {"A": _label("other", "low", "neutral")}  # B has no label
    validation = validate_predictions(tickets, tmp_path)
    metrics = score(tickets, labels, tmp_path, validation)
    assert metrics.scorable_tickets == 1
    assert metrics.missing_label_ids == ["B"]
    assert metrics.exact_match_rate == 1.0  # denominator is scorable tickets only


def test_invalid_prediction_counts_incorrect(tmp_path: Path) -> None:
    tickets = [_ticket("A")]
    _write_pred(tmp_path, "A", category="INVALID", priority="low", sentiment="neutral")
    labels = {"A": _label("other", "low", "neutral")}
    validation = validate_predictions(tickets, tmp_path)
    metrics = score(tickets, labels, tmp_path, validation)
    assert metrics.scorable_tickets == 1
    assert metrics.invalid_predictions == 1
    assert metrics.exact_match_rate == 0.0
    assert metrics.category_accuracy == 0.0


def test_no_scorable_tickets_returns_none(tmp_path: Path) -> None:
    tickets = [_ticket("A")]
    _write_pred(tmp_path, "A", category="other", priority="low", sentiment="neutral")
    labels: dict = {}  # no labels at all
    validation = validate_predictions(tickets, tmp_path)
    metrics = score(tickets, labels, tmp_path, validation)
    assert metrics.scorable_tickets == 0
    assert metrics.category_accuracy is None
    assert metrics.exact_match_rate is None
