"""Retrieval demo CLI.

    python -m rag_support_agent.retrieval.search --query "how do I rotate an API key?"
    python -m rag_support_agent.retrieval.search --query "E_RATE_LIMIT" --show-arms

Prints the hybrid ranking with each arm's contribution, so you can see *why* a source
ranked where it did. ``--show-arms`` also prints the dense-only and BM25-only top lists,
which is the whole point: it makes "hybrid beats either arm alone" visible.

Requires an ingested database (``docker compose up -d`` then ``ingestion.run``).
"""

from __future__ import annotations

import argparse

from rag_support_agent.config import get_settings
from rag_support_agent.knowledge.models import KnowledgeUnit, RetrievalResult
from rag_support_agent.retrieval.hybrid import retrieve_explained


def _fmt(x: float | None, width: int = 6) -> str:
    return f"{x:.3f}".rjust(width) if x is not None else "—".rjust(width)


def _label(unit: KnowledgeUnit) -> str:
    section = unit.section or "(preamble)"
    return f"{unit.source_uri}  ::  {section}"


def _print_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("  (relevance gate returned nothing — no confidently relevant source)")
        return
    print(f"  {'#':>2}  {'rrf':>6}  {'cos':>6} {'d#':>3}  {'bm25':>6} {'s#':>3}  source")
    for i, r in enumerate(results, start=1):
        d_rank = str(r.dense_rank) if r.dense_rank else "—"
        s_rank = str(r.sparse_rank) if r.sparse_rank else "—"
        print(
            f"  {i:>2}  {_fmt(r.score)}  {_fmt(r.dense_similarity)} {d_rank:>3}  "
            f"{_fmt(r.sparse_score)} {s_rank:>3}  {_label(r.unit)}"
        )


def _print_arm(title: str, arm: list[tuple[KnowledgeUnit, float]], top_k: int) -> None:
    print(f"\n  {title}")
    if not arm:
        print("    (no results)")
        return
    for i, (unit, score) in enumerate(arm[:top_k], start=1):
        print(f"    {i:>2}  {_fmt(score)}  {_label(unit)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval demo.")
    parser.add_argument("--query", required=True, help="The question to retrieve for.")
    parser.add_argument("--top-k", type=int, default=None, help="How many results to show.")
    parser.add_argument(
        "--show-arms",
        action="store_true",
        help="Also print the dense-only and BM25-only rankings for comparison.",
    )
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.retrieval_top_k
    results, dense, sparse = retrieve_explained(args.query, top_k=top_k, settings=settings)

    print(f'\nquery: "{args.query}"')
    print(
        f"provider={settings.embedding_provider}  top_k={top_k}  "
        f"gate>=cos {settings.retrieval_min_similarity}  rrf_k={settings.rrf_k}\n"
    )
    print("HYBRID (RRF-fused, gated):")
    _print_results(results)

    if args.show_arms:
        _print_arm("DENSE only (pgvector cosine):", dense, top_k)
        _print_arm("SPARSE only (BM25):", sparse, top_k)
    print()


if __name__ == "__main__":
    main()
