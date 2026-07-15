"""Unit tests for blind-spot detection + observability (M7).

All DB-free and key-free: the clustering is pure over query strings, and the per-request cost
is pure over token counts, so the gap-report logic and the cost wiring are pinned down without
Postgres or an API key in the loop. (The DB-backed persistence + report is exercised live in
the M7 demo, mirroring how M6's freshness DB path is demoed rather than unit-tested.)
"""

import pytest

from rag_support_agent.config import Settings
from rag_support_agent.eval.cost import TokenUsage
from rag_support_agent.generation.answer import _generation_cost, build_answer
from rag_support_agent.generation.generators import ExtractiveGenerator
from rag_support_agent.knowledge.models import AnswerVerdict, KnowledgeUnit, RetrievalResult
from rag_support_agent.observability.blindspot import (
    GapCluster,
    _build_clusters,
    cluster_lexical,
    cluster_semantic,
    salient_terms,
)
from rag_support_agent.retrieval.embeddings import HashEmbedder


# --- salient_terms (the shared retrieval tokenizer) ---------------------------------


def test_salient_terms_drops_stopwords_preserves_codes_and_dedups():
    # Stopwords (how/do/i/the) dropped, the error code survives as one token, "api" deduped.
    terms = salient_terms("How do I fix the E_RATE_LIMIT error api api")
    assert terms == ["fix", "e_rate_limit", "error", "api"]


# --- cluster_lexical (keyless connected components on shared vocabulary) -------------


def test_cluster_lexical_merges_shared_terms_and_splits_disjoint():
    qs = [
        "how do I rotate an api key",     # rotate, api, key
        "revoke an api key immediately",  # revoke, api, key  -> shares api/key with [0]
        "configure webhook retries",      # configure, webhook, retries -> disjoint
    ]
    assert cluster_lexical(qs) == [[0, 1], [2]]


def test_cluster_lexical_is_order_independent():
    # Shuffling the input must not change the partition (membership is a set property). Compare
    # on the query strings, since the indices necessarily differ between the two orderings.
    def partition(qs):
        return {frozenset(qs[i] for i in comp) for comp in cluster_lexical(qs)}

    q1 = ["reset password", "password recovery help", "billing invoice"]
    q2 = ["billing invoice", "password recovery help", "reset password"]
    assert partition(q1) == partition(q2)
    assert partition(q1) == {
        frozenset({"reset password", "password recovery help"}),  # linked by "password"
        frozenset({"billing invoice"}),
    }


def test_cluster_lexical_empty():
    assert cluster_lexical([]) == []


# --- _build_clusters (labelling / counting / examples / sources) --------------------


def test_build_clusters_theme_count_example_sources():
    queries = ["revoke an api key", "rotate the api key now", "configure webhook"]
    sources = ["api-keys.md", None, "webhooks.md"]
    clusters = _build_clusters(queries, sources, cluster_lexical(queries))

    top = clusters[0]
    assert top.count == 2
    assert top.theme == "api key"                # api(2), key(2) win on frequency, alpha-ordered
    assert top.example == "revoke an api key"    # shortest member (most canonical)
    assert top.sources == ["api-keys.md"]        # None filtered out

    tail = clusters[1]
    assert tail == GapCluster(
        theme="configure webhook",
        count=1,
        example="configure webhook",
        terms=["configure", "webhook"],
        sources=["webhooks.md"],
    )


def test_build_clusters_counts_repeated_questions_as_volume():
    # The same gap hit 3x is a bigger gap, not a duplicate to collapse — count reflects volume.
    queries = ["export my data", "export my data", "export my data"]
    clusters = _build_clusters(queries, [None, None, None], cluster_lexical(queries))
    assert len(clusters) == 1 and clusters[0].count == 3


# --- cluster_semantic (embedding-gated; mechanics testable keyless) -----------------


def test_cluster_semantic_merges_by_embedding_similarity():
    # Hash embeddings carry no meaning, but identical strings embed identically (cosine 1.0),
    # so the connected-component mechanics are testable without a key.
    qs = ["rotate api key", "rotate api key", "configure webhook retries"]
    assert cluster_semantic(qs, HashEmbedder(64), threshold=0.9) == [[0, 1], [2]]


# --- per-request cost wiring (pure over token counts) -------------------------------


class _FakeGen:
    def __init__(self, model: str) -> None:
        self.model = model


def test_generation_cost_is_zero_for_keyless_extractive():
    # No tokens spent -> $0 regardless of price band; the keyless default path stays free.
    assert _generation_cost(TokenUsage(0, 0), ExtractiveGenerator()) == 0.0


def test_generation_cost_prices_by_the_model_that_ran():
    m = TokenUsage(1_000_000, 1_000_000)  # 1M in + 1M out
    assert _generation_cost(m, _FakeGen("gpt-4o-mini")) == pytest.approx(0.75)      # 0.15 + 0.60
    assert _generation_cost(m, _FakeGen("gemini-2.5-flash")) == pytest.approx(2.80)  # 0.30 + 2.50


# --- build_answer stays pure (cost/logging are answer_question's job) ----------------


def _answerable(uid: str, dense: float) -> RetrievalResult:
    unit = KnowledgeUnit(id=uid, content=f"To do {uid}, follow these steps.", source_uri=f"{uid}.md")
    return RetrievalResult(unit=unit, score=0.03, dense_similarity=dense)


def test_build_answer_leaves_cost_unset():
    # The pure seam must not price or log — that is the thin answer_question layer's job.
    results = [_answerable("a", 0.80), _answerable("b", 0.20), _answerable("c", 0.18)]
    ans = build_answer("q", results, ExtractiveGenerator(), settings=Settings())
    assert ans.verdict is AnswerVerdict.ANSWERED
    assert ans.cost_usd is None
