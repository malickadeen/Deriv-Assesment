"""Centralized configuration and the allowed enum vocabularies.

Configuration is read from environment variables (loaded from a local ``.env``
by ``python-dotenv``). Secrets are never logged, printed, or written to any
output artifact. The enum tuples defined here are the single source of truth
used by the prompts, the Pydantic models, the validator, and the scorer, so the
allowed vocabulary can never drift between components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Allowed enum vocabularies (single source of truth).
# ---------------------------------------------------------------------------
CATEGORIES: tuple[str, ...] = (
    "payment_issue",
    "account_verification",
    "login_access",
    "trading_problem",
    "other",
)
PRIORITIES: tuple[str, ...] = ("low", "medium", "high")
SENTIMENTS: tuple[str, ...] = ("neutral", "frustrated", "urgent")

# Fields that make up a final, saved prediction artifact — exactly these, no more.
PREDICTION_FIELDS: tuple[str, ...] = (
    "ticket_id",
    "category",
    "priority",
    "sentiment",
    "reasoning",
)

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDER = "openrouter"


@dataclass(frozen=True)
class Config:
    """Immutable run configuration resolved from environment + CLI overrides."""

    mode: str = "mock"  # "real" | "mock"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    prompt_version: str = "v1"

    tickets_path: Path = field(default_factory=lambda: Path("tickets.json"))
    labels_path: Path = field(default_factory=lambda: Path("labels.json"))
    output_dir: Path = field(default_factory=lambda: Path("predictions"))

    validation_path: Path = field(default_factory=lambda: Path("validation.json"))
    metrics_path: Path = field(default_factory=lambda: Path("metrics.json"))
    report_path: Path = field(default_factory=lambda: Path("report.md"))
    log_path: Path = field(default_factory=lambda: Path("llm_calls.jsonl"))

    # Present only in real mode; never logged or reported.
    _api_key: Optional[str] = None

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key


def load_env() -> None:
    """Load ``.env`` into the process environment (idempotent, non-overriding)."""
    load_dotenv(override=False)


def resolve_api_key() -> Optional[str]:
    """Return the OpenRouter API key from the environment, or ``None``."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        key = key.strip()
    return key or None


def build_config(
    *,
    mode: str,
    tickets: Optional[str] = None,
    labels: Optional[str] = None,
    output_dir: Optional[str] = None,
    prompt_version: Optional[str] = None,
) -> Config:
    """Assemble a :class:`Config` from environment variables and CLI overrides.

    The API key is only attached in ``real`` mode; ``mock`` mode never requires
    or reads it. Raises ``ValueError`` if real mode is requested without a key.
    """
    load_env()

    model = os.getenv("MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL

    api_key: Optional[str] = None
    if mode == "real":
        api_key = resolve_api_key()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file to run "
                "in real mode, or use '--mode mock' which requires no API key."
            )

    return Config(
        mode=mode,
        model=model,
        base_url=base_url,
        prompt_version=(prompt_version or "v1"),
        tickets_path=Path(tickets) if tickets else Path("tickets.json"),
        labels_path=Path(labels) if labels else Path("labels.json"),
        output_dir=Path(output_dir) if output_dir else Path("predictions"),
        _api_key=api_key,
    )
