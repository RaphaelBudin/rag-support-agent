"""Pure JSON serialization for the API layer — no FastAPI, no DB.

``serialize_answer`` turns an ``Answer`` (with every trust signal the UI renders) into a
plain JSON-able dict; ``sse_event`` formats one Server-Sent Events frame. Both are pure, so
the API's *data contract* — the exact shape the UI depends on — is unit-tested without a
server or a database in the loop, the same pure-seam discipline as ``build_answer`` and the
eval metric functions.
"""

from __future__ import annotations

import json

from rag_support_agent.knowledge.models import Answer


def serialize_answer(answer: Answer, query: str, provider: str) -> dict:
    """``Answer`` -> the JSON payload the chat UI renders (badge, citations, stale, cost)."""
    return {
        "query": query,
        "provider": provider,
        "verdict": answer.verdict.value,  # "answered" | "abstained"
        "text": answer.text,
        "confidence": answer.confidence,  # 0..1 (retrieval-spread scale, ~0–0.3 typical)
        "citations": [
            {
                "index": c.index,  # the stable inline [n] handle, not a re-enumeration
                "knowledge_unit_id": c.knowledge_unit_id,
                "source_uri": c.source_uri,
                "score": c.score,
            }
            for c in answer.citations
        ],
        "stale_sources": list(answer.stale_sources),  # M6 possibly-stale, cited sources only
        "latency_ms": answer.latency_ms,
        "cost_usd": answer.cost_usd,  # $0 on the keyless extractive path
    }


def sse_event(event: str, data: dict) -> str:
    """One Server-Sent Events frame: an ``event:`` line plus a single JSON ``data:`` line.

    ``ensure_ascii=False`` keeps non-ASCII answer text intact; the trailing blank line is the
    SSE record terminator. The UI's ``EventSource`` dispatches on the ``event:`` name
    (``token`` / ``done`` / ``error``).
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
