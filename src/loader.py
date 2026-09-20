"""Input loading and deterministic input validation.

Loads ``tickets.json`` and ``labels.json`` from configurable paths, validates
their structure independently of any LLM, and reconciles which tickets are
scorable. Ticket IDs, counts, and content are entirely dynamic — nothing here
assumes a specific ID or a fixed number of tickets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from .models import Label, Ticket


class InputError(Exception):
    """Raised when an input file is missing, malformed, or fails validation."""


_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_ticket_id(ticket_id: str) -> str:
    """Return a filesystem-safe stem for a ticket id.

    Prevents path traversal by stripping any directory separators and replacing
    every character outside ``[A-Za-z0-9._-]`` with ``_``. A leading dot is
    also neutralized so ids like ``..`` cannot escape the output directory.
    """
    stem = _UNSAFE_CHARS.sub("_", ticket_id.strip())
    stem = stem.lstrip(".") or "ticket"
    return stem


def prediction_path(output_dir: Path, ticket_id: str) -> Path:
    """Compute the prediction artifact path for a ticket id, safely."""
    return Path(output_dir) / f"{sanitize_ticket_id(ticket_id)}.json"


@dataclass
class LabelReconciliation:
    """Outcome of matching labels against the loaded tickets."""

    labeled_ticket_ids: list[str] = field(default_factory=list)
    missing_label_ids: list[str] = field(default_factory=list)  # tickets w/o labels
    extra_label_ids: list[str] = field(default_factory=list)  # labels w/o tickets


def _read_json(path: Path):
    if not path.exists():
        raise InputError(f"Input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}") from exc


def load_tickets(path: Path) -> list[Ticket]:
    """Load and validate tickets. Enforces the array schema, required fields,
    non-empty IDs, and uniqueness of ticket IDs."""
    data = _read_json(path)
    if not isinstance(data, list):
        raise InputError(f"{path} must contain a top-level JSON array of tickets.")
    if not data:
        raise InputError(f"{path} contains no tickets.")

    tickets: list[Ticket] = []
    seen: set[str] = set()
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise InputError(f"Ticket at index {index} must be a JSON object.")
        try:
            ticket = Ticket.model_validate(raw)
        except ValidationError as exc:
            raise InputError(
                f"Ticket at index {index} failed validation: {exc.errors()}"
            ) from exc
        if ticket.ticket_id in seen:
            raise InputError(f"Duplicate ticket_id detected: {ticket.ticket_id!r}")
        seen.add(ticket.ticket_id)
        tickets.append(ticket)
    return tickets


def load_labels(path: Path) -> dict[str, Label]:
    """Load and validate labels. Enforces the object schema and enum values."""
    data = _read_json(path)
    if not isinstance(data, dict):
        raise InputError(f"{path} must contain a top-level JSON object keyed by ticket_id.")

    labels: dict[str, Label] = {}
    for ticket_id, raw in data.items():
        if not isinstance(raw, dict):
            raise InputError(f"Label for {ticket_id!r} must be a JSON object.")
        try:
            labels[ticket_id] = Label.model_validate(raw)
        except ValidationError as exc:
            raise InputError(
                f"Label for {ticket_id!r} failed validation: {exc.errors()}"
            ) from exc
    return labels


def reconcile_labels(
    tickets: list[Ticket], labels: dict[str, Label]
) -> LabelReconciliation:
    """Determine which tickets have labels, which are missing, and which label
    entries are extra (present in labels but not in tickets).

    Policy: a ticket must have a ground-truth label to be *scorable*. Missing
    labels are reported and their tickets are treated as unscorable; extra
    labels are reported as warnings and otherwise ignored.
    """
    ticket_ids = [t.ticket_id for t in tickets]
    ticket_id_set = set(ticket_ids)
    label_id_set = set(labels.keys())

    labeled = [tid for tid in ticket_ids if tid in label_id_set]
    missing = [tid for tid in ticket_ids if tid not in label_id_set]
    extra = sorted(label_id_set - ticket_id_set)

    return LabelReconciliation(
        labeled_ticket_ids=labeled,
        missing_label_ids=missing,
        extra_label_ids=extra,
    )
