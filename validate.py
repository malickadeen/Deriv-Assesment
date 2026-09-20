"""Standalone artifact validation command.

Inspects the artifacts produced by a pipeline run and returns a nonzero exit
code if any required check fails. This does NOT call the LLM and does not trust
the LLM to validate anything. It also confirms that the deterministic scoring
implementation exists in ``src/scorer.py`` and is wired into ``run.py`` — it
does not try to "prove" the numbers by inspecting their values.

Usage:
    python validate.py
    python validate.py --tickets tickets.json --labels labels.json --output-dir predictions
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.loader import (
    InputError,
    load_labels,
    load_tickets,
    prediction_path,
    reconcile_labels,
)
from src.logger import read_log
from src.validator import validate_prediction_file

REQUIRED_LOG_FIELDS = (
    "stage", "ticket_id", "timestamp", "provider", "model",
    "prompt_version", "output_artifact",
)


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks: int = 0

    def check(self, condition: bool, ok_msg: str, fail_msg: str) -> bool:
        self.checks += 1
        if condition:
            print(f"  [PASS] {ok_msg}")
        else:
            print(f"  [FAIL] {fail_msg}")
            self.failures.append(fail_msg)
        return condition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate.py", description="Validate pipeline artifacts after a run."
    )
    parser.add_argument("--tickets", default="tickets.json")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--report", default="report.md")
    parser.add_argument("--log", default="llm_calls.jsonl")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    c = Checker()

    tickets_path = Path(args.tickets)
    labels_path = Path(args.labels)
    output_dir = Path(args.output_dir)

    print("== Artifact presence ==")
    c.check(tickets_path.exists(), f"{tickets_path} exists", f"{tickets_path} missing")
    c.check(labels_path.exists(), f"{labels_path} exists", f"{labels_path} missing")
    c.check(output_dir.is_dir(), f"{output_dir}/ exists", f"{output_dir}/ missing")
    for name in (args.validation, args.metrics, args.report, args.log):
        p = Path(name)
        c.check(p.exists(), f"{name} exists", f"{name} missing")

    print("== Scoring implementation ==")
    scorer = Path("src/scorer.py")
    c.check(scorer.exists(), "src/scorer.py present", "src/scorer.py missing")
    run_src = Path("run.py").read_text(encoding="utf-8") if Path("run.py").exists() else ""
    c.check(
        "from src.scorer import" in run_src and "score(" in run_src,
        "run.py imports and uses src.scorer.score",
        "run.py does not wire in src.scorer.score",
    )

    # Load inputs to drive per-ticket checks.
    try:
        tickets = load_tickets(tickets_path)
        labels = load_labels(labels_path)
    except InputError as exc:
        print(f"  [FAIL] Could not load inputs: {exc}")
        c.failures.append(f"input load: {exc}")
        _summary(c)
        return 1

    recon = reconcile_labels(tickets, labels)
    if recon.missing_label_ids:
        print(f"  [WARN] tickets without labels: {', '.join(recon.missing_label_ids)}")
    if recon.extra_label_ids:
        print(f"  [WARN] labels without tickets: {', '.join(recon.extra_label_ids)}")

    print("== Per-ticket prediction artifacts ==")
    for ticket in tickets:
        path = prediction_path(output_dir, ticket.ticket_id)
        result = validate_prediction_file(path, ticket.ticket_id)
        # A documented failure artifact still satisfies "artifact exists".
        exists = path.exists()
        c.check(
            exists,
            f"{ticket.ticket_id}: artifact present",
            f"{ticket.ticket_id}: no prediction or failure artifact at {path}",
        )
        if exists and not result.valid:
            print(f"    [WARN] {ticket.ticket_id} invalid: {result.errors}")

    print("== LLM call log ==")
    log_records = read_log(Path(args.log))
    c.check(bool(log_records), "llm_calls.jsonl has records", "llm_calls.jsonl is empty")
    fields_ok = all(
        all(f in rec for f in REQUIRED_LOG_FIELDS) for rec in log_records
    ) if log_records else False
    c.check(fields_ok, "all log records contain required fields",
            "some log records are missing required fields")

    logged_ids = {rec.get("ticket_id") for rec in log_records}
    every_ticket_logged = all(t.ticket_id in logged_ids for t in tickets)
    c.check(every_ticket_logged, "every ticket has >=1 classification log record",
            "some tickets have no classification log record")

    print("== validation.json / metrics.json shape ==")
    try:
        vjson = json.loads(Path(args.validation).read_text(encoding="utf-8"))
        c.check(
            {"total_tickets", "valid_outputs", "invalid_outputs", "results"} <= set(vjson),
            "validation.json has required keys", "validation.json missing keys",
        )
    except (OSError, json.JSONDecodeError) as exc:
        c.check(False, "", f"validation.json unreadable: {exc}")
    try:
        mjson = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
        c.check(
            {"total_tickets", "scorable_tickets", "category_accuracy",
             "exact_match_rate", "per_ticket"} <= set(mjson),
            "metrics.json has required keys", "metrics.json missing keys",
        )
    except (OSError, json.JSONDecodeError) as exc:
        c.check(False, "", f"metrics.json unreadable: {exc}")

    return _summary(c)


def _summary(c: Checker) -> int:
    print("\n== Summary ==")
    passed = c.checks - len(c.failures)
    print(f"{passed}/{c.checks} checks passed.")
    if c.failures:
        print("FAILED checks:")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print("All artifact checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
