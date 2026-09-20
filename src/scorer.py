"""Deterministic scoring of validated predictions against ground truth.

All metrics are computed here in plain Python — the LLM never calculates or
supplies them. Scoring policy:

* A ticket is *scorable* only if it has a ground-truth label. Tickets without a
  label are excluded from the denominator and reported separately.
* An invalid prediction counts as incorrect on every axis for a scorable ticket
  (it is never treated as an exact match). It is not silently dropped.
* Denominator = number of scorable tickets. If zero tickets are scorable, all
  accuracy metrics are ``None`` (documented representation instead of 0/0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .loader import prediction_path
from .models import Label, Ticket
from .validator import ValidationReport, load_valid_prediction


@dataclass
class TicketScore:
    ticket_id: str
    scorable: bool
    valid_prediction: bool
    category_correct: bool
    priority_correct: bool
    sentiment_correct: bool
    exact_match: bool
    note: str = ""


@dataclass
class Metrics:
    total_tickets: int
    scorable_tickets: int
    invalid_predictions: int
    category_accuracy: Optional[float]
    priority_accuracy: Optional[float]
    sentiment_accuracy: Optional[float]
    exact_match_rate: Optional[float]
    per_ticket: list[TicketScore] = field(default_factory=list)
    missing_label_ids: list[str] = field(default_factory=list)
    extra_label_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_tickets": self.total_tickets,
            "scorable_tickets": self.scorable_tickets,
            "invalid_predictions": self.invalid_predictions,
            "category_accuracy": self.category_accuracy,
            "priority_accuracy": self.priority_accuracy,
            "sentiment_accuracy": self.sentiment_accuracy,
            "exact_match_rate": self.exact_match_rate,
            "missing_label_ids": self.missing_label_ids,
            "extra_label_ids": self.extra_label_ids,
            "per_ticket": [asdict(s) for s in self.per_ticket],
        }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def score(
    tickets: list[Ticket],
    labels: dict[str, Label],
    output_dir: Path,
    validation: ValidationReport,
    *,
    extra_label_ids: Optional[list[str]] = None,
) -> Metrics:
    """Compute per-ticket scores and aggregate metrics."""
    valid_ids = {r.ticket_id for r in validation.results if r.valid}

    per_ticket: list[TicketScore] = []
    missing_label_ids: list[str] = []
    scorable = 0
    invalid_predictions = 0
    cat_ok = pri_ok = sen_ok = exact_ok = 0

    for ticket in tickets:
        tid = ticket.ticket_id
        label = labels.get(tid)

        if label is None:
            missing_label_ids.append(tid)
            per_ticket.append(
                TicketScore(
                    ticket_id=tid, scorable=False, valid_prediction=tid in valid_ids,
                    category_correct=False, priority_correct=False,
                    sentiment_correct=False, exact_match=False,
                    note="No ground-truth label; not scorable.",
                )
            )
            continue

        scorable += 1
        prediction = None
        if tid in valid_ids:
            prediction = load_valid_prediction(prediction_path(output_dir, tid), tid)

        if prediction is None:
            invalid_predictions += 1
            per_ticket.append(
                TicketScore(
                    ticket_id=tid, scorable=True, valid_prediction=False,
                    category_correct=False, priority_correct=False,
                    sentiment_correct=False, exact_match=False,
                    note="Invalid prediction; counted as incorrect.",
                )
            )
            continue

        c = prediction.category == label.category
        p = prediction.priority == label.priority
        s = prediction.sentiment == label.sentiment
        e = c and p and s
        cat_ok += c
        pri_ok += p
        sen_ok += s
        exact_ok += e
        per_ticket.append(
            TicketScore(
                ticket_id=tid, scorable=True, valid_prediction=True,
                category_correct=c, priority_correct=p, sentiment_correct=s,
                exact_match=e,
            )
        )

    return Metrics(
        total_tickets=len(tickets),
        scorable_tickets=scorable,
        invalid_predictions=invalid_predictions,
        category_accuracy=_ratio(cat_ok, scorable),
        priority_accuracy=_ratio(pri_ok, scorable),
        sentiment_accuracy=_ratio(sen_ok, scorable),
        exact_match_rate=_ratio(exact_ok, scorable),
        per_ticket=per_ticket,
        missing_label_ids=missing_label_ids,
        extra_label_ids=extra_label_ids or [],
    )
