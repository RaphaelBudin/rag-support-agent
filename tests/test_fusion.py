"""Unit tests for the retrieval building blocks that need no DB.

Fusion, tokenization, and the relevance gate are pure functions — the parts most
worth pinning down — so they're tested here without Postgres in the loop.
"""

from rag_support_agent.knowledge.models import KnowledgeUnit, RetrievalResult
from rag_support_agent.retrieval.hybrid import (
    _passes_gate,
    _tokenize,
    reciprocal_rank_fusion,
)


def _unit(uid: str) -> KnowledgeUnit:
    return KnowledgeUnit(id=uid, content=uid, source_uri=f"{uid}.md")


def test_rrf_orders_by_summed_reciprocal_rank():
    a, b, c = _unit("a"), _unit("b"), _unit("c")
    dense = [(a, 0.9), (b, 0.1)]      # a@1, b@2
    sparse = [(b, 5.0), (c, 4.0)]     # b@1, c@2
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    order = [u.id for u, _ in fused]
    # b appears at rank 1 and rank 2 -> highest; a@1 beats c@2.
    assert order == ["b", "a", "c"]
    scores = {u.id: s for u, s in fused}
    assert scores["b"] == 1 / 61 + 1 / 62
    assert scores["a"] == 1 / 61
    assert scores["c"] == 1 / 62


def test_rrf_ignores_raw_score_magnitude():
    # A huge BM25 score must not outrank a better *position* in the other arm.
    a, b = _unit("a"), _unit("b")
    dense = [(a, 0.99)]               # a@1
    sparse = [(b, 10_000.0)]          # b@1 — same rank, wildly bigger raw score
    fused = dict((u.id, s) for u, s in reciprocal_rank_fusion([dense, sparse]))
    assert fused["a"] == fused["b"]   # rank decides, not magnitude


def test_tokenizer_preserves_error_codes_and_api_tokens():
    tokens = _tokenize("Error E_RATE_LIMIT on POST /v1/sessions with ak_test_abc")
    assert "e_rate_limit" in tokens        # not shredded into e/rate/limit
    assert "ak_test_abc" in tokens
    assert "v1" in tokens and "sessions" in tokens


def _result(dense=None, sparse=None) -> RetrievalResult:
    return RetrievalResult(unit=_unit("x"), score=0.0, dense_similarity=dense, sparse_score=sparse)


def test_gate_keeps_strong_dense_match():
    assert _passes_gate(_result(dense=0.42), min_similarity=0.15)


def test_gate_keeps_exact_keyword_hit_even_with_weak_dense():
    assert _passes_gate(_result(dense=0.02, sparse=3.1), min_similarity=0.15)


def test_gate_drops_when_both_arms_are_weak():
    assert not _passes_gate(_result(dense=0.02, sparse=None), min_similarity=0.15)
    assert not _passes_gate(_result(dense=None, sparse=None), min_similarity=0.15)
