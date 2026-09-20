"""Main pipeline entry point.

Executes the stages strictly in order:

    LOAD_INPUTS -> CLASSIFY -> VALIDATE -> SCORE -> REPORT

Validation is never skipped before scoring, and invalid predictions are handled
explicitly (counted as incorrect, never silently treated as valid).

Usage:
    python run.py                      # default: mock mode
    python run.py --mode real          # OpenRouter / GPT-4o-mini
    python run.py --mode mock          # offline deterministic classifier
    python run.py --tickets t.json --labels l.json --output-dir predictions
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from src.classifier import build_backend, classify_ticket
from src.config import build_config
from src.loader import InputError, load_labels, load_tickets, reconcile_labels
from src.logger import LLMCallLogger
from src.reporter import build_report, write_report
from src.scorer import score
from src.validator import validate_predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Replayable AI evaluation pipeline for support-ticket classification.",
    )
    parser.add_argument(
        "--mode", choices=["real", "mock"], default="mock",
        help="'real' calls OpenRouter (needs OPENROUTER_API_KEY); "
             "'mock' runs a deterministic offline classifier. Default: mock.",
    )
    parser.add_argument("--tickets", default=None, help="Path to tickets.json (default: tickets.json).")
    parser.add_argument("--labels", default=None, help="Path to labels.json (default: labels.json).")
    parser.add_argument("--output-dir", default=None, help="Predictions directory (default: predictions/).")
    parser.add_argument(
        "--prompt-version", default=None, choices=["v1", "v3"],
        help="Initial prompt version: 'v1' (baseline) or 'v3' (rubric + few-shot). "
             "Default: v1. The retry always uses the strict 'v2' prompt.",
    )
    return parser.parse_args(argv)


def _reset_predictions_dir(output_dir: Path) -> None:
    """Clear only the generated predictions directory before a run.

    This keeps stale predictions from mixing with the current evaluation. Input
    files are never touched.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- Config -----------------------------------------------------------
    try:
        config = build_config(
            mode=args.mode,
            tickets=args.tickets,
            labels=args.labels,
            output_dir=args.output_dir,
            prompt_version=args.prompt_version,
        )
    except ValueError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    print(f"Mode: {config.mode} | Model: {config.model} | Prompt: {config.prompt_version}")

    # ---- Stage 1: LOAD_INPUTS --------------------------------------------
    try:
        tickets = load_tickets(config.tickets_path)
        labels = load_labels(config.labels_path)
    except InputError as exc:
        print(f"[input error] {exc}", file=sys.stderr)
        return 2

    recon = reconcile_labels(tickets, labels)
    print(f"Loaded {len(tickets)} ticket(s), {len(labels)} label(s).")
    if recon.missing_label_ids:
        print(f"  ! Missing labels (unscorable): {', '.join(recon.missing_label_ids)}")
    if recon.extra_label_ids:
        print(f"  ! Extra labels (ignored):      {', '.join(recon.extra_label_ids)}")

    # ---- Stage 2: CLASSIFY ------------------------------------------------
    _reset_predictions_dir(config.output_dir)
    try:
        backend = build_backend(config)
    except (ValueError, ImportError) as exc:
        print(f"[backend error] {exc}", file=sys.stderr)
        return 2

    logger = LLMCallLogger(config.log_path)
    print("Classifying tickets...")
    for ticket in tickets:
        try:
            outcome = classify_ticket(backend, ticket, config, logger)
        except Exception as exc:  # never let one ticket crash the whole run
            print(f"  {ticket.ticket_id}: unexpected error ({type(exc).__name__}); continuing.")
            continue
        status = "ok" if outcome.success else f"FAILED ({outcome.error_type})"
        print(f"  {ticket.ticket_id}: {status} [{outcome.attempts} attempt(s)]")

    # ---- Stage 3: VALIDATE ------------------------------------------------
    validation = validate_predictions(tickets, config.output_dir)
    config.validation_path.write_text(
        json.dumps(validation.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Validation: {validation.valid_outputs} valid / {validation.invalid_outputs} invalid.")

    # ---- Stage 4: SCORE ---------------------------------------------------
    metrics = score(
        tickets, labels, config.output_dir, validation,
        extra_label_ids=recon.extra_label_ids,
    )
    config.metrics_path.write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "Metrics: "
        f"cat={metrics.category_accuracy} pri={metrics.priority_accuracy} "
        f"sen={metrics.sentiment_accuracy} exact={metrics.exact_match_rate} "
        f"(scorable={metrics.scorable_tickets})"
    )

    # ---- Stage 5: REPORT --------------------------------------------------
    report = build_report(
        mode=config.mode,
        model=config.model,
        prompt_version=config.prompt_version,
        validation=validation,
        metrics=metrics,
    )
    write_report(config.report_path, report)
    print(f"Report written to {config.report_path}")

    # Nonzero exit if any output was invalid, so CI can react.
    return 0 if validation.invalid_outputs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
