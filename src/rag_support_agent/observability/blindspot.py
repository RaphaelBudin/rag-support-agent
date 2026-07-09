"""Blind-spot detection + observability.

Every low-confidence or no-source query is logged. Aggregated, these become a
knowledge-gap report: "the top things users ask that we can't answer well." That
closes the loop — it tells you exactly which docs to write next.

TODO(M7): persist events + aggregate the gap report; log cost/latency per request.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class QueryEvent:
    query: str
    confidence: float
    abstained: bool
    top_source: str | None
    latency_ms: float
    cost_usd: float
    at: datetime | None = None


def record(event: QueryEvent) -> None:
    """Persist a query event for later analysis."""
    raise NotImplementedError


def knowledge_gap_report(limit: int = 10) -> list[tuple[str, int]]:
    """Top unanswered/low-confidence query clusters -> (theme, count)."""
    raise NotImplementedError
