"""Prompt templates and prompt versioning.

All prompt strings live here so they never scatter across modules. Two versions
are provided:

* ``v1`` — the default classification prompt.
* ``v2`` — the stricter "repair" prompt used on the single retry after an
  invalid response. It restates the enums and forbids any non-JSON output.

Both are built from the same enum vocabularies in :mod:`src.config`, so the
allowed values can never drift from what the validator enforces.
"""

from __future__ import annotations

from .config import CATEGORIES, PRIORITIES, SENTIMENTS
from .models import Ticket

# v1: baseline classification prompt.
# v2: stricter "repair" prompt used on the single retry after an invalid response.
# v3: rubric + few-shot variant, selectable as an initial prompt to compare
#     against v1 (retry always falls back to the strict v2).
PROMPT_VERSIONS: tuple[str, ...] = ("v1", "v2", "v3")

# Compact, schema-shaped exemplars for v3. These are illustrative decision cues,
# NOT drawn from tickets.json/labels.json, so they never leak evaluation data.
FEW_SHOT_EXAMPLES = """\
Example A (explicit time pressure + money loss => urgent, high):
  message: "My order is stuck and I'm losing money every second, fix it now."
  -> {"category": "trading_problem", "priority": "high", "sentiment": "urgent"}
Example B (calm informational request, no impact => neutral, low):
  message: "Just wondering where I download my monthly statement. No rush."
  -> {"category": "other", "priority": "low", "sentiment": "neutral"}
Example C (repeated failures + blocked access, annoyed but not time-critical => frustrated, high):
  message: "Third time my document was rejected and I still can't access my account."
  -> {"category": "account_verification", "priority": "high", "sentiment": "frustrated"}"""

CATEGORY_DEFINITIONS = """\
- payment_issue: Payments, charges, deposits, withdrawals, duplicate charges,
  failed payments, or transaction debits.
- account_verification: Identity verification, KYC, document submission,
  verification requirements, or verification-related restrictions.
- login_access: Login, password resets, expired reset links, account access,
  or authentication problems.
- trading_problem: Trading activity, orders, market execution, trading errors,
  or trading functionality.
- other: Issues that do not reasonably fit any category above."""

PRIORITY_DEFINITIONS = """\
- high: Substantial customer impact, blocks important functionality, involves
  repeated failures, or clearly requires prompt attention per the ticket.
- medium: Affects the customer but does not clearly indicate severe or
  immediate impact.
- low: Informational, minor, or limited impact.
Do NOT automatically mark every payment or verification issue as high. Base the
priority strictly on the actual ticket content; do not invent facts."""

SENTIMENT_DEFINITIONS = """\
- neutral: Neutral or informational tone, no clear frustration or urgency.
- frustrated: Dissatisfaction, repeated failed attempts, inconvenience, or
  explicit frustration.
- urgent: A strong, explicit need for immediate attention.
Do NOT infer urgency solely from the presence of a payment or account issue.
Use the customer's actual wording and context."""

_SCHEMA_BLOCK = f"""\
Return a single JSON object with EXACTLY these fields:
{{
  "ticket_id": "<the ticket's id, unchanged>",
  "category": one of {list(CATEGORIES)},
  "priority": one of {list(PRIORITIES)},
  "sentiment": one of {list(SENTIMENTS)},
  "reasoning": "<one short, human-readable sentence>"
}}"""


def _ticket_block(ticket: Ticket) -> str:
    return (
        f"ticket_id: {ticket.ticket_id}\n"
        f"channel: {ticket.channel}\n"
        f"subject: {ticket.subject}\n"
        f"message: {ticket.message}"
    )


def system_prompt(version: str = "v1") -> str:
    """Return the system prompt for the given version."""
    base = f"""\
You are a customer-support ticket classifier for a trading/fintech platform.
Classify each ticket into a category, priority, and sentiment using ONLY the
definitions below. You never evaluate or grade your own answer, and you never
receive the correct answer.

CATEGORIES:
{CATEGORY_DEFINITIONS}

PRIORITIES:
{PRIORITY_DEFINITIONS}

SENTIMENTS:
{SENTIMENT_DEFINITIONS}

{_SCHEMA_BLOCK}

Output rules:
- Respond with the JSON object ONLY.
- Do NOT wrap the JSON in Markdown code fences.
- Do NOT add any text before or after the JSON.
- Keep "reasoning" short (one sentence)."""
    if version == "v2":
        base += """

STRICT MODE: Your previous response could not be parsed or failed validation.
Respond with exactly ONE valid JSON object, all five fields present, every enum
value spelled exactly as listed, no code fences, and no surrounding text."""
    elif version == "v3":
        base += f"""

DECISION RUBRIC (apply in order, using only the ticket's wording):
- sentiment: choose "urgent" ONLY on explicit immediacy or active loss
  ("now", "immediately", "losing money"); choose "frustrated" on repeated
  failures or clear dissatisfaction; otherwise "neutral".
- priority: choose "high" only on blocked access, repeated failures, or stated
  urgency; "low" for informational/no-impact requests; otherwise "medium".
- Do not escalate priority or sentiment just because the topic is payments,
  verification, or trading.

WORKED EXAMPLES (format only; do not copy their values blindly):
{FEW_SHOT_EXAMPLES}"""
    return base


def user_prompt(ticket: Ticket, version: str = "v1") -> str:
    """Return the user prompt (the ticket to classify) for the given version."""
    return f"Classify this ticket:\n{_ticket_block(ticket)}"


def build_messages(ticket: Ticket, version: str = "v1") -> list[dict[str, str]]:
    """Build a role/content message list for the chat model."""
    return [
        {"role": "system", "content": system_prompt(version)},
        {"role": "user", "content": user_prompt(ticket, version)},
    ]
