"""Generation demo CLI (M3).

    python -m rag_support_agent.generation.ask --query "How do I rotate an API key?"
    LLM_PROVIDER=gemini python -m rag_support_agent.generation.ask --query "..."

Retrieves (hybrid + gate), synthesizes a grounded answer, and prints it with its inline
citations and verdict. Keyless by default (``LLM_PROVIDER=extractive``) so it runs with no
API key; set ``LLM_PROVIDER=gemini`` for real synthesis.

Requires an ingested database (``docker compose up -d`` then ``ingestion.run``).
"""

from __future__ import annotations

import argparse

from rag_support_agent.config import get_settings
from rag_support_agent.generation.answer import answer_question
from rag_support_agent.knowledge.models import AnswerVerdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Grounded answer generation demo.")
    parser.add_argument("--query", required=True, help="The question to answer.")
    parser.add_argument("--top-k", type=int, default=None, help="Passages to retrieve.")
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Don't append this query to the blind-spot / observability log (M7).",
    )
    args = parser.parse_args()

    settings = get_settings()
    answer = answer_question(
        args.query, top_k=args.top_k, settings=settings, record_event=not args.no_log
    )

    print(f'\nquery: "{args.query}"')
    print(
        f"provider={settings.llm_provider}  verdict={answer.verdict.value}  "
        f"confidence={answer.confidence:.3f}"
        + (f"  latency={answer.latency_ms:.0f}ms" if answer.latency_ms is not None else "")
        + (f"  cost=${answer.cost_usd:.6f}" if answer.cost_usd is not None else "")
    )
    print("\n" + answer.text + "\n")

    if answer.verdict is AnswerVerdict.ANSWERED:
        print("citations:")
        if not answer.citations:
            print("  (none — answer cited no passage)")
        for c in answer.citations:
            print(f"  [{c.index}] {c.source_uri}  (unit {c.knowledge_unit_id}, rrf {c.score:.4f})")

    if answer.stale_sources:
        print("\n⚠ possibly stale sources (age is a decay-risk proxy, not a verdict — re-verify):")
        for src in answer.stale_sources:
            print(f"  - {src}")
    print()


if __name__ == "__main__":
    main()
