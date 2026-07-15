"""Knowledge-gap report + observability panel (M7 demo).

    python -m rag_support_agent.observability.gap_report --limit 10
    python -m rag_support_agent.observability.gap_report --mode lexical      # force keyless
    EMBEDDING_PROVIDER=gemini python -m rag_support_agent.observability.gap_report  # semantic

Reads the blind-spot log (seeded by ``observability.replay`` or the ``ask`` CLI) and prints:

  1. the top unanswered-query themes — "the docs to write next" — clustered keyless-lexical
     or embedding-semantic depending on the configured provider (``--mode`` forces one);
  2. a per-request observability rollup (cost / latency / tokens) over *all* logged traffic.
"""

from __future__ import annotations

import argparse

from rag_support_agent.config import get_settings
from rag_support_agent.observability.blindspot import gap_clusters, observability_summary


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _ms(x: float | None) -> str:
    return f"{x:.0f} ms" if x is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge-gap report + observability panel.")
    parser.add_argument("--limit", type=int, default=10, help="Top-N gap themes to show.")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "lexical", "semantic"],
        help="Clusterer: auto (semantic if an embedder is set, else lexical) | lexical | semantic.",
    )
    args = parser.parse_args()

    settings = get_settings()
    clusters = gap_clusters(args.limit, settings=settings, mode=args.mode)
    summary = observability_summary(settings=settings)

    effective = args.mode
    if effective == "auto":
        effective = "semantic" if settings.embedding_provider != "hash" else "lexical"

    print("\n=== Knowledge-gap report — top things asked that we can't answer ===")
    print(
        f"(clustering: {effective}; abstained {summary.n_abstained} / {summary.n_events} logged "
        "queries)\n"
    )
    if not clusters:
        print("  (no abstained queries logged yet — run observability.replay first)")
    for i, c in enumerate(clusters, start=1):
        near = ", ".join(c.sources) if c.sources else "— (nothing cleared the relevance gate)"
        print(f"  {i}. [{c.count}x] theme: {c.theme}")
        print(f'       e.g. "{c.example}"')
        print(f"       nearest source: {near}")

    print("\n=== Observability (per request, over all logged traffic) ===")
    print(f"  requests            {summary.n_events}")
    print(f"  abstain rate        {_pct(summary.abstain_rate)}")
    print(f"  p50 / p95 latency   {_ms(summary.p50_latency_ms)} / {_ms(summary.p95_latency_ms)}")
    if summary.mean_input_tokens is not None:
        print(
            f"  mean tokens/query   {summary.mean_input_tokens:.0f} in / "
            f"{summary.mean_output_tokens:.0f} out"
        )
    print(f"  mean cost/query     ${(summary.mean_cost_usd or 0.0):.6f}")
    print(f"  total cost          ${summary.total_cost_usd:.6f}\n")


if __name__ == "__main__":
    main()
