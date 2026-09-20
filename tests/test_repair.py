"""Tests for deterministic JSON extraction/repair (no LLM involved)."""

from __future__ import annotations

import pytest

from src.repair import extract_first_json_object, parse_prediction_json, strip_code_fences


def test_plain_json_parses() -> None:
    obj = parse_prediction_json('{"ticket_id": "T-1", "priority": "low"}')
    assert obj["ticket_id"] == "T-1"
    assert obj["priority"] == "low"


def test_fenced_json_parses() -> None:
    raw = '```json\n{"ticket_id": "T-2", "category": "other"}\n```'
    obj = parse_prediction_json(raw)
    assert obj["ticket_id"] == "T-2"
    assert obj["category"] == "other"


def test_json_with_surrounding_prose() -> None:
    raw = 'Sure! Here is the result:\n{"ticket_id": "T-3", "priority": "high"} Thanks!'
    obj = parse_prediction_json(raw)
    assert obj["ticket_id"] == "T-3"


def test_nested_braces_in_strings() -> None:
    raw = '{"ticket_id": "T-4", "reasoning": "uses {curly} braces"}'
    obj = parse_prediction_json(raw)
    assert obj["reasoning"] == "uses {curly} braces"


def test_strip_code_fences_no_fence() -> None:
    assert strip_code_fences('{"a": 1}') == '{"a": 1}'


def test_extract_first_object_only() -> None:
    raw = '{"first": 1} {"second": 2}'
    assert extract_first_json_object(raw) == '{"first": 1}'


def test_invalid_response_raises() -> None:
    with pytest.raises(ValueError):
        parse_prediction_json("no json at all here")


def test_unbalanced_braces_raises() -> None:
    with pytest.raises(ValueError):
        parse_prediction_json('{"ticket_id": "T-5"')
