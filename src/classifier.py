"""LLM classification: real (OpenRouter via LangChain) and mock backends.

Both backends expose the same ``generate(ticket, version) -> str`` contract and
run through the same attempt loop: one initial call, deterministic JSON parse +
Pydantic validation, and exactly one stricter retry on failure. The loop never
fabricates a valid prediction to hide a failure — a ticket that cannot be
classified after the retry is recorded as a failure artifact.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from pydantic import ValidationError

from .config import PROVIDER, Config
from .loader import prediction_path
from .logger import LLMCallLogger
from .models import Prediction, Ticket
from .prompts import build_messages
from .repair import parse_prediction_json


class ClassificationError(Exception):
    """Raised when a single generate() call cannot produce usable text."""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend(Protocol):
    """A text-generating backend for classification."""

    def generate(self, ticket: Ticket, version: str) -> str:  # pragma: no cover
        ...


class RealBackend:
    """OpenRouter-backed chat model via LangChain's OpenAI-compatible client."""

    def __init__(self, config: Config) -> None:
        # Imported lazily so mock mode never requires the dependency at import.
        from langchain_openai import ChatOpenAI

        if not config.api_key:
            raise ValueError("Real backend requires an OpenRouter API key.")
        self._llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=0,
            timeout=60,
            max_retries=0,  # retries are orchestrated explicitly by this module
            # Nudge structured output where the provider/model supports it.
            model_kwargs={"response_format": {"type": "json_object"}},
        )

    def generate(self, ticket: Ticket, version: str) -> str:
        messages = build_messages(ticket, version)
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:  # network/provider errors
            raise ClassificationError(f"LLM call failed: {type(exc).__name__}") from exc
        content = getattr(response, "content", None)
        if isinstance(content, list):  # some providers return content parts
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ClassificationError("LLM returned empty content.")
        return content


class MockBackend:
    """Deterministic, rule-based classifier for offline runs.

    Uses only the ticket's own text (never labels.json). Rules are documented
    and intentionally simple; mock output is NOT representative of real LLM
    performance.
    """

    CATEGORY_KEYWORDS = {
        "payment_issue": (
            "charge", "charged", "payment", "deposit", "withdraw", "refund",
            "debit", "transaction", "card", "billing", "invoice",
        ),
        "account_verification": (
            "verif", "kyc", "identity", "passport", "document", "id ",
            "proof", "selfie",
        ),
        "login_access": (
            "login", "log in", "password", "reset", "access", "locked",
            "authenticat", "sign in", "2fa", "otp",
        ),
        "trading_problem": (
            "trade", "trading", "order", "market", "execute", "execution",
            "position", "buy", "sell", "price",
        ),
    }
    HIGH_KEYWORDS = (
        "urgent", "immediately", "asap", "as soon as possible", "losing money",
        "cannot access", "can't access", "blocked", "third time", "again",
        "twice", "duplicate", "stuck",
    )
    LOW_KEYWORDS = (
        "no rush", "just curious", "just wondering", "whenever", "question about",
        "when you get a chance",
    )
    URGENT_KEYWORDS = ("urgent", "immediately", "asap", "right now", "losing money")
    FRUSTRATED_KEYWORDS = (
        "frustrat", "again", "third time", "keeps", "still not", "annoyed",
        "disappointed", "twice", "duplicate",
    )

    def generate(self, ticket: Ticket, version: str) -> str:
        text = f"{ticket.subject}\n{ticket.message}".lower()

        category = "other"
        best = 0
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best:
                best, category = hits, cat

        if any(kw in text for kw in self.LOW_KEYWORDS):
            priority = "low"
        elif any(kw in text for kw in self.HIGH_KEYWORDS):
            priority = "high"
        else:
            priority = "medium"

        if any(kw in text for kw in self.URGENT_KEYWORDS):
            sentiment = "urgent"
        elif any(kw in text for kw in self.FRUSTRATED_KEYWORDS):
            sentiment = "frustrated"
        else:
            sentiment = "neutral"

        return json.dumps(
            {
                "ticket_id": ticket.ticket_id,
                "category": category,
                "priority": priority,
                "sentiment": sentiment,
                "reasoning": "Deterministic mock rule-based classification.",
            }
        )


def build_backend(config: Config) -> Backend:
    """Instantiate the backend for the configured mode."""
    if config.mode == "real":
        return RealBackend(config)
    return MockBackend()


# ---------------------------------------------------------------------------
# Attempt loop
# ---------------------------------------------------------------------------
@dataclass
class TicketOutcome:
    """Result of classifying one ticket, for use by the orchestrator/report."""

    ticket_id: str
    success: bool
    attempts: int
    artifact_path: Path
    error_type: Optional[str] = None


def _attempt(
    backend: Backend, ticket: Ticket, version: str
) -> tuple[Optional[Prediction], Optional[str]]:
    """Run one generate + parse + validate. Returns (prediction, error_type)."""
    try:
        raw = backend.generate(ticket, version)
    except ClassificationError as exc:
        return None, f"api_error:{exc}"

    try:
        obj = parse_prediction_json(raw)
    except ValueError as exc:
        return None, f"invalid_json:{exc}"

    # Enforce that the model did not relabel a different ticket.
    if obj.get("ticket_id") != ticket.ticket_id:
        obj["ticket_id"] = ticket.ticket_id

    try:
        prediction = Prediction.model_validate(obj)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ())) or "schema"
        return None, f"schema_error:{loc}"

    return prediction, None


def classify_ticket(
    backend: Backend,
    ticket: Ticket,
    config: Config,
    logger: LLMCallLogger,
) -> TicketOutcome:
    """Classify a single ticket with one initial call and one retry.

    Saves a valid prediction artifact on success, or a failure artifact (with an
    ``error`` field, never a fabricated label) on failure. Logs each attempt.
    """
    artifact = prediction_path(config.output_dir, ticket.ticket_id)
    versions = [config.prompt_version, "v2"]  # initial, then stricter retry

    last_error: Optional[str] = None
    for i, version in enumerate(versions, start=1):
        start = time.perf_counter()
        prediction, error = _attempt(backend, ticket, version)
        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.log_attempt(
            ticket_id=ticket.ticket_id,
            attempt=i,
            success=prediction is not None,
            provider=PROVIDER,
            model=config.model,
            mode=config.mode,
            prompt_version=version,
            output_artifact=str(artifact),
            latency_ms=latency_ms,
            error_type=error,
        )

        if prediction is not None:
            _write_json(artifact, prediction.model_dump())
            return TicketOutcome(
                ticket_id=ticket.ticket_id,
                success=True,
                attempts=i,
                artifact_path=artifact,
            )
        last_error = error

    # Both attempts failed — persist an explicit failure artifact.
    _write_json(
        artifact,
        {
            "ticket_id": ticket.ticket_id,
            "error": last_error or "unknown_error",
            "valid": False,
        },
    )
    return TicketOutcome(
        ticket_id=ticket.ticket_id,
        success=False,
        attempts=len(versions),
        artifact_path=artifact,
        error_type=last_error,
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
