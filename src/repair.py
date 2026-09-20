"""Deterministic JSON extraction from noisy LLM responses.

This module contains NO LLM calls. It extracts the first balanced JSON object
from a raw text response — handling Markdown code fences and leading/trailing
prose — and parses it. It does not judge whether the *content* is correct; the
Pydantic schema and the validator do that afterward.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def strip_code_fences(text: str) -> str:
    """Remove a surrounding Markdown code fence (```json ... ```), if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (e.g. "```json" or "```").
    lines = lines[1:]
    # Drop the closing fence line if present.
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_first_json_object(text: str) -> Optional[str]:
    """Return the substring of the first balanced top-level ``{...}`` object.

    Brace matching is string-aware: braces inside JSON string literals (and
    escaped quotes) are ignored. Returns ``None`` if no balanced object exists.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_prediction_json(raw: str) -> dict[str, Any]:
    """Best-effort deterministic parse of a raw model response into a dict.

    Strategy (no LLM involved):
      1. Try to parse the whole (fence-stripped) string as JSON.
      2. Fall back to extracting the first balanced JSON object and parsing it.

    Raises ``ValueError`` if no valid JSON object can be recovered.
    """
    if raw is None:
        raise ValueError("Empty response: no content to parse.")

    candidate = strip_code_fences(raw)

    # Attempt 1: the cleaned string is itself a JSON object.
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the first balanced object from anywhere in the text.
    snippet = extract_first_json_object(candidate) or extract_first_json_object(raw)
    if snippet is None:
        raise ValueError("No JSON object found in the response.")
    try:
        parsed = json.loads(snippet)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Extracted text is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Extracted JSON is not an object.")
    return parsed
