"""JSONL logging of every LLM classification attempt.

One JSON object per line is appended to ``llm_calls.jsonl``. Secrets, API keys,
and full raw responses are never written. Each attempt (including retries) is
logged as its own record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class LLMCallLogger:
    """Appends structured records for each LLM attempt to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        # Fresh log per run so stale attempts never mix with the current run.
        self.path.write_text("", encoding="utf-8")

    def log_attempt(
        self,
        *,
        ticket_id: str,
        attempt: int,
        success: bool,
        provider: str,
        model: str,
        mode: str,
        prompt_version: str,
        output_artifact: str,
        response_format: str = "json_object",
        latency_ms: Optional[float] = None,
        error_type: Optional[str] = None,
        stage: str = "classify",
    ) -> None:
        """Append one attempt record. Never records secrets or raw responses."""
        record: dict[str, Any] = {
            "stage": stage,
            "ticket_id": ticket_id,
            "timestamp": utc_now_iso(),
            "provider": provider,
            "model": model,
            "mode": mode,
            "prompt_version": prompt_version,
            "attempt": attempt,
            "success": success,
            "response_format": response_format,
            "output_artifact": output_artifact,
        }
        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 2)
        if error_type:
            record["error_type"] = error_type

        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL log file into a list of records (skips blank lines)."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records
