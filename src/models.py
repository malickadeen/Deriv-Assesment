"""Pydantic schemas for tickets, ground-truth labels, and predictions.

These models enforce structural and enum-level correctness. They are used both
for structured-output enforcement on the LLM side and for deterministic
validation on the scoring side. Cross-record checks (e.g. that a prediction's
ticket_id matches its source ticket) are intentionally *not* handled here — the
validator performs those separately, because a single-record model cannot know
about the ticket it is supposed to correspond to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import CATEGORIES, PRIORITIES, SENTIMENTS

CategoryLiteral = Literal[
    "payment_issue",
    "account_verification",
    "login_access",
    "trading_problem",
    "other",
]
PriorityLiteral = Literal["low", "medium", "high"]
SentimentLiteral = Literal["neutral", "frustrated", "urgent"]


class Ticket(BaseModel):
    """A single customer-support ticket from ``tickets.json``."""

    model_config = ConfigDict(extra="ignore")

    ticket_id: str
    subject: str
    message: str
    channel: str

    @field_validator("ticket_id")
    @classmethod
    def ticket_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("ticket_id must be a non-empty string")
        return v


class Label(BaseModel):
    """A ground-truth label entry from ``labels.json``."""

    model_config = ConfigDict(extra="ignore")

    category: CategoryLiteral
    priority: PriorityLiteral
    sentiment: SentimentLiteral


class Prediction(BaseModel):
    """A strict LLM classification result.

    ``extra="forbid"`` guarantees the saved artifact contains exactly the
    required fields and nothing else.
    """

    model_config = ConfigDict(extra="forbid")

    ticket_id: str = Field(min_length=1)
    category: CategoryLiteral
    priority: PriorityLiteral
    sentiment: SentimentLiteral
    reasoning: str

    @field_validator("reasoning")
    @classmethod
    def reasoning_is_string(cls, v: str) -> str:
        # Pydantic already enforces str type; keep a short, human-readable value.
        return v.strip()


# Re-exported so callers can build prompts/messages without re-importing config.
ALLOWED = {
    "category": CATEGORIES,
    "priority": PRIORITIES,
    "sentiment": SENTIMENTS,
}
