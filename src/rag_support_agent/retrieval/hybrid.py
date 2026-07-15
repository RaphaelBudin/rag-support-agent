"""Hybrid retrieval: dense (pgvector) + sparse (BM25), fused with RRF.

Pure vector search misses exact-match tokens that matter in support — error codes
(`E_RATE_LIMIT`), API names (`POST /v1/sessions`), key prefixes (`ak_test_`). BM25
nails those; vectors catch paraphrase ("stop a key working" -> "revoke a key"). We
run both arms and combine them with **Reciprocal Rank Fusion**, which fuses by *rank*
rather than *score* — so we never have to normalize a cosine similarity (~0..1)
against a BM25 score (unbounded, corpus-dependent) or tune a per-query weight.

The public entrypoint is ``retrieve``. A relevance gate then drops candidates that
neither arm found convincing, which is what lets an out-of-scope question surface
nothing strong (feeding abstention in M4).
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.knowledge.db import get_conn
from rag_support_agent.knowledge.models import KnowledgeUnit, RetrievalResult
from rag_support_agent.retrieval.embeddings import Embedder, get_embedder

# Keep error codes / API tokens intact: `[a-z0-9_]+` means `E_RATE_LIMIT` and
# `ak_test_` survive lowercasing as single tokens instead of being shredded into
# `e`, `rate`, `limit`. That single choice is *why* BM25 wins on exact matches.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Drop a small set of English function words before BM25. Without this, BM25 scores
# documents on "how/the/a" overlap — which hurts precision AND lets a fully
# out-of-scope query ("how do I train a model") leak past the relevance gate on
# stopword matches alone. Content words only => zero real overlap scores zero.
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in into is it its of on "
    "or that the their this to view was what when where which who why will with you "
    "your".split()
)

_UNIT_COLS = (
    "id, content, source_uri, section, chunk_index, "
    "content_hash, source_updated_at, created_at"
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on the error-code-preserving pattern, drop stopwords.

    Public because it is the repo's single notion of a *salient term*: the M7 blind-spot
    gap report clusters unanswered queries on the very same tokens BM25 ranks on, so
    "the questions retrieval failed on" are grouped by the vocabulary the retriever indexes.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


_tokenize = tokenize  # internal alias; the sparse-arm callers below keep their name.


def _row_to_unit(row: tuple) -> KnowledgeUnit:
    (id_, content, source_uri, section, chunk_index, content_hash,
     source_updated_at, created_at) = row
    return KnowledgeUnit(
        id=id_,
        content=content,
        source_uri=source_uri,
        section=section,
        chunk_index=chunk_index,
        content_hash=content_hash,
        source_updated_at=source_updated_at,
        created_at=created_at,
    )


def _load_all_units(conn) -> list[KnowledgeUnit]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_UNIT_COLS} FROM knowledge_units")
        return [_row_to_unit(r) for r in cur.fetchall()]


def vector_search(
    conn, embedder: Embedder, query: str, top_k: int
) -> list[tuple[KnowledgeUnit, float]]:
    """Cosine top-k over pgvector. Returns (unit, cosine_similarity) pairs.

    The query is embedded with the *same* provider that ingested the docs — mixing
    embedders would compare vectors from different spaces and return noise. We use the
    ``<=>`` cosine-distance operator (it matches the HNSW index's ``vector_cosine_ops``
    opclass, so the index is actually used) and report ``1 - distance`` as similarity.
    """
    qvec = np.asarray(embedder.embed([query])[0], dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {_UNIT_COLS}, 1 - (embedding <=> %s) AS similarity
            FROM knowledge_units
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (qvec, qvec, top_k),
        )
        rows = cur.fetchall()
    return [(_row_to_unit(r[:-1]), float(r[-1])) for r in rows]


def bm25_search(conn, query: str, top_k: int) -> list[tuple[KnowledgeUnit, float]]:
    """Keyword top-k via BM25 (IDF + length normalization). Returns (unit, score).

    BM25 is built in-memory over the whole corpus. That is honest ("real" BM25, unlike
    Postgres ``ts_rank`` which is TF-only) and instant at this scale. What breaks when
    it scales: rebuilding the index per query is O(N) — at a large corpus you push
    sparse retrieval into the DB (Postgres FTS / a dedicated search index) or persist
    the index. The section heading is folded into the searchable text so an exact
    heading term ("Rotating a key") counts.
    """
    units = _load_all_units(conn)
    if not units:
        return []
    corpus = [_tokenize(f"{u.section or ''} {u.content}") for u in units]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(units, scores), key=lambda us: us[1], reverse=True)
    # Drop zero-overlap docs: a 0 score means none of the query's terms appear.
    return [(u, float(s)) for u, s in ranked[:top_k] if s > 0]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[KnowledgeUnit, float]]], k: int = 60
) -> list[tuple[KnowledgeUnit, float]]:
    """Fuse multiple rankings via RRF: score = sum over lists of 1 / (k + rank).

    Rank is 1-based. Only position matters — the per-arm scores are ignored — so no
    normalization and no per-query weight tuning. ``k`` (default 60, the TREC default)
    dampens how much the very top ranks dominate.
    """
    fused: dict[str, float] = {}
    units: dict[str, KnowledgeUnit] = {}
    for ranking in rankings:
        for rank, (unit, _score) in enumerate(ranking, start=1):
            fused[unit.id] = fused.get(unit.id, 0.0) + 1.0 / (k + rank)
            units[unit.id] = unit
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(units[uid], score) for uid, score in ordered]


def _passes_gate(result: RetrievalResult, min_similarity: float) -> bool:
    """Relevance gate: a candidate survives if *either* arm found it convincing.

    Dense clears an absolute cosine floor (semantically close), OR sparse has any
    positive BM25 score (an exact keyword hit — legitimate evidence in support, where
    a matching error code is a strong signal). Failing both means out-of-scope.

    Deliberately coarse: it thresholds the *dense* magnitude but not the BM25 magnitude
    (scale problem), so it is a pre-filter, not a trust decision. Calibrated abstention
    is M4's job, computed from score spread + grounding — a different threshold.
    """
    if result.dense_similarity is not None and result.dense_similarity >= min_similarity:
        return True
    return result.sparse_score is not None and result.sparse_score > 0


def _assemble(
    fused: list[tuple[KnowledgeUnit, float]],
    dense: list[tuple[KnowledgeUnit, float]],
    sparse: list[tuple[KnowledgeUnit, float]],
) -> list[RetrievalResult]:
    """Attach each arm's rank + raw score to the fused ranking."""
    dense_idx = {u.id: (i + 1, sc) for i, (u, sc) in enumerate(dense)}
    sparse_idx = {u.id: (i + 1, sc) for i, (u, sc) in enumerate(sparse)}
    results: list[RetrievalResult] = []
    for unit, fscore in fused:
        d = dense_idx.get(unit.id)
        sp = sparse_idx.get(unit.id)
        results.append(
            RetrievalResult(
                unit=unit,
                score=fscore,
                dense_similarity=d[1] if d else None,
                sparse_score=sp[1] if sp else None,
                dense_rank=d[0] if d else None,
                sparse_rank=sp[0] if sp else None,
            )
        )
    return results


def _hybrid(
    conn, embedder: Embedder, query: str, top_k: int, settings: Settings
) -> tuple[list[RetrievalResult], list, list]:
    """Core hybrid retrieve. Returns (gated_results, dense_arm, sparse_arm)."""
    pool = max(settings.retrieval_candidate_pool, top_k)
    dense = vector_search(conn, embedder, query, pool)
    sparse = bm25_search(conn, query, pool)
    fused = reciprocal_rank_fusion([dense, sparse], k=settings.rrf_k)
    results = _assemble(fused, dense, sparse)
    gated = [r for r in results if _passes_gate(r, settings.retrieval_min_similarity)]
    return gated[:top_k], dense, sparse


def retrieve(
    query: str, top_k: int | None = None, settings: Settings | None = None
) -> list[RetrievalResult]:
    """Public entrypoint: hybrid retrieve + relevance gate (drop weak matches)."""
    s = settings or get_settings()
    top_k = top_k or s.retrieval_top_k
    conn = get_conn(s)
    try:
        gated, _, _ = _hybrid(conn, get_embedder(s), query, top_k, s)
    finally:
        conn.close()
    return gated


def retrieve_explained(
    query: str, top_k: int | None = None, settings: Settings | None = None
) -> tuple[list[RetrievalResult], list, list]:
    """Like ``retrieve`` but also returns the raw dense-only and sparse-only arms,
    so the demo/CLI can show *why* hybrid ranks differently from either arm alone."""
    s = settings or get_settings()
    top_k = top_k or s.retrieval_top_k
    conn = get_conn(s)
    try:
        return _hybrid(conn, get_embedder(s), query, top_k, s)
    finally:
        conn.close()
