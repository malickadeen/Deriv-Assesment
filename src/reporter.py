"""Markdown report generation from the actual run artifacts.

The report is assembled deterministically from validation + scoring outputs.
Metrics are never recomputed by an LLM. Error analysis is derived from observed
validation errors and per-axis mismatches; each observation is paired with a
concrete, non-speculative improvement.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .scorer import Metrics
from .validator import ValidationReport


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _mode_banner(mode: str) -> str:
    if mode == "mock":
        return (
            "> **Mock mode.** Predictions were produced by a deterministic, "
            "rule-based classifier — NOT by GPT-4o-mini. These results are not "
            "representative of real LLM performance."
        )
    return "> **Real mode.** Predictions were produced by the configured OpenRouter model."


def _error_analysis(validation: ValidationReport, metrics: Metrics) -> str:
    lines: list[str] = []

    # Observed validation error types.
    error_counter: Counter[str] = Counter()
    for result in validation.results:
        for err in result.errors:
            key = err.split(":")[0].strip()
            error_counter[key] += 1

    if error_counter:
        lines.append("**Observed validation errors:**\n")
        for key, count in error_counter.most_common():
            lines.append(f"- {key} — {count} occurrence(s)")
        lines.append("")

    # Per-axis mismatches among scorable, validly-predicted tickets.
    cat_miss = pri_miss = sen_miss = 0
    for s in metrics.per_ticket:
        if s.scorable and s.valid_prediction:
            cat_miss += not s.category_correct
            pri_miss += not s.priority_correct
            sen_miss += not s.sentiment_correct

    lines.append("**Classification mismatches (valid predictions only):**\n")
    lines.append(f"- Category mismatches: {cat_miss}")
    lines.append(f"- Priority mismatches: {pri_miss}")
    lines.append(f"- Sentiment mismatches: {sen_miss}")
    lines.append("")

    if metrics.missing_label_ids:
        lines.append(
            f"**Missing ground-truth labels:** {len(metrics.missing_label_ids)} "
            f"ticket(s) — {', '.join(metrics.missing_label_ids)} (excluded from scoring)."
        )
    if metrics.extra_label_ids:
        lines.append(
            f"**Extra label entries (no matching ticket):** "
            f"{', '.join(metrics.extra_label_ids)} (ignored, reported as warning)."
        )
    if metrics.invalid_predictions:
        lines.append(
            f"**Invalid predictions:** {metrics.invalid_predictions} scorable ticket(s) "
            "had an unusable prediction and were counted as incorrect."
        )

    return "\n".join(lines).strip()


def _improvements(metrics: Metrics, validation: ValidationReport) -> list[str]:
    """Pick 2–4 concrete, evidence-backed improvements for this run."""
    suggestions: list[str] = []

    pri_miss = sum(
        1 for s in metrics.per_ticket
        if s.scorable and s.valid_prediction and not s.priority_correct
    )
    sen_miss = sum(
        1 for s in metrics.per_ticket
        if s.scorable and s.valid_prediction and not s.sentiment_correct
    )
    cat_miss = sum(
        1 for s in metrics.per_ticket
        if s.scorable and s.valid_prediction and not s.category_correct
    )

    if pri_miss:
        suggestions.append(
            "**Priority:** Observed priority mismatches. Improvement — add 2–3 "
            "borderline low/medium/high examples to the prompt and sharpen the "
            "medium-vs-high boundary so payment/verification issues are not "
            "over-escalated."
        )
    if sen_miss:
        suggestions.append(
            "**Sentiment:** Observed sentiment mismatches. Improvement — clarify "
            "the frustrated-vs-urgent distinction with wording cues (e.g. explicit "
            "time pressure => urgent) rather than topic-based inference."
        )
    if cat_miss:
        suggestions.append(
            "**Category:** Observed category mismatches. Improvement — add "
            "disambiguating notes for overlapping cases (e.g. a failed payment "
            "during a trade) so the dominant intent is chosen consistently."
        )
    if metrics.invalid_predictions or validation.invalid_outputs:
        suggestions.append(
            "**Output validity:** Some outputs failed schema validation. "
            "Improvement — keep enforcing JSON-object response format and the "
            "single stricter retry; consider few-shot JSON exemplars in the prompt."
        )

    if not suggestions:
        suggestions = [
            "**Coverage:** Metrics look strong on this set. Improvement — expand "
            "the evaluation set with more borderline tickets to stress the "
            "priority and sentiment boundaries.",
            "**Robustness:** Add adversarial/malformed inputs to confirm the "
            "retry/repair path and validator behave as intended.",
        ]
    return suggestions[:4]


def build_report(
    *,
    mode: str,
    model: str,
    prompt_version: str,
    validation: ValidationReport,
    metrics: Metrics,
) -> str:
    """Render the full Markdown report."""
    ts = datetime.now(timezone.utc).isoformat()

    per_ticket_rows = []
    score_by_id = {s.ticket_id: s for s in metrics.per_ticket}
    for result in validation.results:
        s = score_by_id.get(result.ticket_id)
        if s is None:
            continue
        scorable = "yes" if s.scorable else "no"
        valid = "yes" if result.valid else "no"
        cat = "-" if not s.scorable else ("✓" if s.category_correct else "✗")
        pri = "-" if not s.scorable else ("✓" if s.priority_correct else "✗")
        sen = "-" if not s.scorable else ("✓" if s.sentiment_correct else "✗")
        exact = "-" if not s.scorable else ("✓" if s.exact_match else "✗")
        per_ticket_rows.append(
            f"| {result.ticket_id} | {valid} | {scorable} | {cat} | {pri} | {sen} | {exact} |"
        )

    lines = [
        "# AI Ticket Classification — Evaluation Report",
        "",
        _mode_banner(mode),
        "",
        "## Run Summary",
        "",
        f"- Generated: `{ts}`",
        f"- Mode: `{mode}`",
        f"- Model: `{model}`",
        f"- Prompt version: `{prompt_version}`",
        f"- Tickets processed: **{validation.total_tickets}**",
        f"- Valid outputs: **{validation.valid_outputs}**",
        f"- Invalid outputs: **{validation.invalid_outputs}**",
        f"- Scorable tickets: **{metrics.scorable_tickets}**",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Category accuracy | {_pct(metrics.category_accuracy)} |",
        f"| Priority accuracy | {_pct(metrics.priority_accuracy)} |",
        f"| Sentiment accuracy | {_pct(metrics.sentiment_accuracy)} |",
        f"| Exact-match rate | {_pct(metrics.exact_match_rate)} |",
        "",
        "_Denominator = scorable tickets (those with a ground-truth label). "
        "Invalid predictions count as incorrect._",
        "",
        "## Per-Ticket Summary",
        "",
        "| Ticket | Valid | Scorable | Category | Priority | Sentiment | Exact |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *per_ticket_rows,
        "",
        "## Error Analysis",
        "",
        _error_analysis(validation, metrics) or "_No errors observed._",
        "",
        "## Suggested Improvements",
        "",
        *[f"{i}. {s}" for i, s in enumerate(_improvements(metrics, validation), start=1)],
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")
