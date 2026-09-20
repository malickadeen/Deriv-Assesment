# AI Ticket Classification — Evaluation Pipeline

A replayable, production-minded command-line pipeline that classifies customer
support tickets with an LLM (GPT-4o-mini via OpenRouter) and evaluates the
results with **deterministic Python** — validation, scoring, and reporting are
never delegated to the model.

## 1. Overview

Given a set of support tickets and a set of ground-truth labels, the pipeline:

1. Loads and validates the inputs.
2. Sends each ticket to an LLM for classification into
   `category` / `priority` / `sentiment` (+ a short `reasoning`).
3. Parses and validates every response deterministically, with one stricter
   retry on failure.
4. Saves one prediction artifact per ticket.
5. Scores predictions against ground truth in plain Python.
6. Writes a Markdown evaluation report, a JSON metrics file, a validation file,
   and a JSONL log of every LLM attempt.

It works with any ticket IDs, any ticket count, and any labels file that follow
the documented schema — nothing is hardcoded to the sample data.

## 2. Pipeline stages

```
LOAD_INPUTS -> CLASSIFY -> VALIDATE -> SCORE -> REPORT
```

Validation is never skipped before scoring. Invalid predictions are handled
explicitly (counted as incorrect) and never silently treated as valid.

## 3. Project structure

```
.
├── tickets.json          # input: array of tickets
├── labels.json           # input: ground-truth labels keyed by ticket_id
├── .env / .env.example   # OpenRouter key + model (secrets never committed)
├── requirements.txt
├── run.py                # main orchestrator (the 5 stages)
├── validate.py           # standalone artifact-validation command
├── src/
│   ├── config.py         # env config + allowed enum vocabularies
│   ├── models.py         # Pydantic schemas (Ticket, Label, Prediction)
│   ├── loader.py         # input loading + deterministic input validation
│   ├── prompts.py        # versioned prompts (v1 + v2 stricter retry)
│   ├── classifier.py     # real (OpenRouter) + mock backends, attempt loop
│   ├── repair.py         # deterministic JSON extraction (no LLM)
│   ├── validator.py      # deterministic prediction validation
│   ├── scorer.py         # deterministic metrics
│   ├── reporter.py       # Markdown report generation
│   └── logger.py         # JSONL LLM-call logging
├── predictions/          # output: one {ticket_id}.json per ticket
├── validation.json       # output: per-ticket validation results
├── metrics.json          # output: computed metrics
├── report.md             # output: human-readable report
├── llm_calls.jsonl       # output: one JSON record per LLM attempt
└── tests/                # pytest suite (no API key required)
```

## 4. Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

(The repo already ships with a `venv/`; you can also just use that.)

## 5. Environment configuration

The API key is loaded from `.env` via `python-dotenv` — **never** hardcode it.
Copy the example and fill in your key:

```bash
cp .env.example .env
```

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
MODEL=openai/gpt-4o-mini
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # optional override
```

Secrets are never printed, logged, or written to any artifact. Mock mode needs
no key.

### OpenRouter setup

1. Create a key at <https://openrouter.ai/keys>.
2. Put it in `.env` as `OPENROUTER_API_KEY`.
3. The pipeline uses LangChain's OpenAI-compatible `ChatOpenAI` client pointed
   at `https://openrouter.ai/api/v1` with model `openai/gpt-4o-mini`.

## 6. Usage

### Real mode (OpenRouter / GPT-4o-mini)

```bash
python run.py --mode real
```

### Mock mode (offline, no API key)

```bash
python run.py --mode mock
```

`python run.py` with no `--mode` defaults to **mock** (safe, offline, no key).

### Optional overrides

```bash
python run.py \
    --mode real \
    --tickets custom_tickets.json \
    --labels custom_labels.json \
    --output-dir predictions \
    --prompt-version v1
```

### Validate the generated artifacts

```bash
python validate.py
```

### Run the tests

```bash
pytest
```

## 7. Input schemas

**tickets.json** — a JSON array:

```json
[{ "ticket_id": "T-1001", "subject": "...", "message": "...", "channel": "email" }]
```

Validated for: valid JSON, top-level array, required non-empty `ticket_id`,
required `subject` / `message` / `channel`, and unique ticket IDs. Channel
values are not restricted to any fixed set.

**labels.json** — a JSON object keyed by `ticket_id`:

```json
{ "T-1001": { "category": "payment_issue", "priority": "high", "sentiment": "frustrated" } }
```

Validated for: valid JSON, top-level object, required `category` / `priority` /
`sentiment`, and allowed enum values.

### Allowed enum values

- **category**: `payment_issue`, `account_verification`, `login_access`,
  `trading_problem`, `other`
- **priority**: `low`, `medium`, `high`
- **sentiment**: `neutral`, `frustrated`, `urgent`

## 8. Output artifacts (what each file is)

| File | Meaning |
| --- | --- |
| `labels.json` | **Input** ground truth. Never shown to the model. |
| `predictions/{id}.json` | **Model output** per ticket (or a failure record). |
| `validation.json` | Deterministic per-ticket validation results. |
| `metrics.json` | Computed accuracy / exact-match metrics. |
| `report.md` | Human-readable evaluation report. |
| `llm_calls.jsonl` | One record per LLM attempt (including retries). |

The model never sees `labels.json` and never scores its own answers.

## 9. Retry / repair strategy

- One initial classification call with prompt **v1**.
- The raw response is cleaned of Markdown fences and parsed by a deterministic
  JSON extractor (`src/repair.py`) that pulls the first balanced `{...}` object
  — **no LLM is used to repair JSON**.
- The parsed object is validated against the Pydantic `Prediction` schema.
- On invalid JSON **or** schema failure, exactly **one** retry runs with prompt
  **v2** (stricter: restates the enums, forbids code fences and extra text).
- Both attempts are logged separately.
- If the retry also fails, a **failure artifact** (`{"valid": false, "error": ...}`)
  is written. A failed result is **never** replaced with a fabricated label.

## 10. Scoring policy

- The denominator is the number of **scorable** tickets = tickets that have a
  ground-truth label.
- Tickets with **missing labels** are not scorable; they are reported and
  excluded from the denominator (never silently mis-scored).
- **Extra labels** (a label with no matching ticket) are reported as warnings
  and ignored.
- An **invalid prediction** for a scorable ticket counts as incorrect on every
  axis and is never treated as an exact match — it is not dropped.
- If **no** tickets are scorable, accuracy metrics are `null` (documented
  representation instead of dividing by zero).

Metrics: category accuracy, priority accuracy, sentiment accuracy, exact-match
rate (all three correct), plus per-ticket pass/fail. All computed in
`src/scorer.py`.

## 11. Replayability & output handling

- The `predictions/` directory is cleared at the start of each run so stale
  predictions never mix with the current evaluation. **Input files are never
  modified.**
- `llm_calls.jsonl` is rewritten fresh each run.
- Every input ticket gets a prediction artifact (valid or a failure record).

## 12. Mock mode details

Mock mode uses a deterministic, documented keyword rule-based classifier
(`MockBackend` in `src/classifier.py`). It uses only the ticket's own text —
**never** `labels.json`. Its outputs run through the same validation, scoring,
and reporting path, and its log records carry `"mode": "mock"`. The report
clearly states that mock results are **not** representative of real LLM
performance, and outputs are not presented as GPT-4o-mini output.

## 13. Prompt versioning

Prompts live only in `src/prompts.py`:

- **v1** — baseline classification prompt (default initial prompt).
- **v2** — stricter "repair" prompt used only on the single retry.
- **v3** — a rubric + few-shot variant: adds an ordered decision rubric for
  priority/sentiment and three schema-shaped worked examples (illustrative, not
  drawn from the evaluation data). Selectable as an initial prompt to A/B it
  against v1.

The chosen initial version is recorded in every `llm_calls.jsonl` record and in
`report.md`. Pick it with `--prompt-version {v1,v3}`; the retry always falls
back to the strict v2.

```bash
# A/B two prompt versions in real mode (save each run's metrics to compare)
python run.py --mode real --prompt-version v1 && cp metrics.json metrics_v1.json
python run.py --mode real --prompt-version v3 && cp metrics.json metrics_v3.json
```

On the shipped 5-ticket sample, v1 and v3 produce identical results
(category 100%, priority 80%, sentiment 80%, exact 60%): the two misses are
borderline label-judgment cases, not prompt weaknesses, so the rubric does not
change them. The mechanism is in place for larger sets where it can help.

## 14. Limitations

- Mock mode's rule-based accuracy is illustrative only.
- Real-mode results depend on OpenRouter/model availability and are subject to
  normal LLM variability (temperature is pinned to 0 to reduce it).
- Structured-output nudging uses `response_format=json_object`; the
  deterministic parser + validator are the real guarantee, not the SDK.

## 15. Example commands

```bash
# Offline smoke test (no key needed)
python run.py --mode mock
python validate.py
pytest

# Real evaluation against OpenRouter
python run.py --mode real
python validate.py

# Custom inputs
python run.py --mode real --tickets my_tickets.json --labels my_labels.json
```
