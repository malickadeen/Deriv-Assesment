"""Deterministic validation of prediction artifacts.

Validation is performed entirely by Python code — never by the LLM. Each
prediction file is checked for existence, valid JSON, object shape, required
non-null fields, correct types, allowed enum values, and — crucially — that its
``ticket_id`` matches the source ticket it was produced for (a cross-record
check that a single-record Pydantic model cannot perform on its own).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from .config import CATEGORIES, PREDICTION_FIELDS, PRIORITIES, SENTIMENTS
from .loader import prediction_path
from .models import Prediction, Ticket


@dataclass
class TicketValidation:
    ticket_id: str
    valid: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    total_tickets: int
    valid_outputs: int
    invalid_outputs: int
    results: list[TicketValidation]

    def to_dict(self) -> dict:
        return {
            "total_tickets": self.total_tickets,
            "valid_outputs": self.valid_outputs,
            "invalid_outputs": self.invalid_outputs,
            "results": [asdict(r) for r in self.results],
        }


def validate_prediction_file(path: Path, expected_ticket_id: str) -> TicketValidation:
    """Validate one prediction artifact against its expected source ticket."""
    errors: list[str] = []

    if not path.exists():
        return TicketValidation(expected_ticket_id, False, ["Prediction file does not exist."])

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return TicketValidation(expected_ticket_id, False, [f"Cannot read file: {exc}"])

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return TicketValidation(expected_ticket_id, False, [f"Invalid JSON: {exc}"])

    if not isinstance(data, dict):
        return TicketValidation(expected_ticket_id, False, ["Prediction is not a JSON object."])

    # Reject explicit failure artifacts up front (documented failure records).
    if data.get("valid") is False or "error" in data:
        msg = data.get("error", "Marked invalid by classifier.")
        return TicketValidation(expected_ticket_id, False, [f"Classifier failure artifact: {msg}"])

    # Required fields present and non-null.
    for fieldname in PREDICTION_FIELDS:
        if fieldname not in data:
            errors.append(f"Missing required field: {fieldname}")
        elif data[fieldname] is None:
            errors.append(f"Field is null: {fieldname}")

    # ticket_id type + cross-record match against the source ticket.
    tid = data.get("ticket_id")
    if tid is not None and not isinstance(tid, str):
        errors.append("ticket_id must be a string.")
    elif isinstance(tid, str) and tid != expected_ticket_id:
        errors.append(f"ticket_id mismatch: expected {expected_ticket_id!r}, got {tid!r}")

    # Enum checks (explicit, human-readable messages for debugging).
    if isinstance(data.get("category"), str) and data["category"] not in CATEGORIES:
        errors.append(f"Invalid category: {data['category']}")
    if isinstance(data.get("priority"), str) and data["priority"] not in PRIORITIES:
        errors.append(f"Invalid priority: {data['priority']}")
    if isinstance(data.get("sentiment"), str) and data["sentiment"] not in SENTIMENTS:
        errors.append(f"Invalid sentiment: {data['sentiment']}")

    if data.get("reasoning") is not None and not isinstance(data.get("reasoning"), str):
        errors.append("reasoning must be a string.")

    # Final structural gate via Pydantic (also forbids extra fields).
    if not errors:
        try:
            Prediction.model_validate(data)
        except ValidationError as exc:
            errors.append(f"Schema validation failed: {exc.errors()}")

    return TicketValidation(expected_ticket_id, len(errors) == 0, errors)


def validate_predictions(
    tickets: list[Ticket], output_dir: Path
) -> ValidationReport:
    """Validate the prediction artifact for every input ticket."""
    results: list[TicketValidation] = []
    for ticket in tickets:
        path = prediction_path(output_dir, ticket.ticket_id)
        results.append(validate_prediction_file(path, ticket.ticket_id))

    valid = sum(1 for r in results if r.valid)
    return ValidationReport(
        total_tickets=len(tickets),
        valid_outputs=valid,
        invalid_outputs=len(tickets) - valid,
        results=results,
    )


def load_valid_prediction(path: Path, expected_ticket_id: str) -> Optional[Prediction]:
    """Return a validated :class:`Prediction`, or ``None`` if the file is invalid."""
    result = validate_prediction_file(path, expected_ticket_id)
    if not result.valid:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Prediction.model_validate(data)
