"""Hybrid retrieval: dense (pgvector) + sparse (BM25), fused.

Pure vector search misses exact-match tokens that matter in support — error
codes, API names, flag names. BM25 catches those; vectors catch paraphrase.
Reciprocal rank fusion combines both without tuning a weight per query.

TODO(M2): implement against the schema created by ingestion.
"""

from __future__ import annotations

from rag_support_agent.knowledge.models import KnowledgeUnit


def vector_search(query: str, top_k: int) -> list[tuple[KnowledgeUnit, float]]:
    """Cosine top-k over pgvector. Returns (unit, score) pairs."""
    raise NotImplementedError


def bm25_search(query: str, top_k: int) -> list[tuple[KnowledgeUnit, float]]:
    """Keyword top-k (BM25). Returns (unit, score) pairs."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    rankings: list[list[tuple[KnowledgeUnit, float]]], k: int = 60
) -> list[tuple[KnowledgeUnit, float]]:
    """Fuse multiple rankings via RRF: score = sum(1 / (k + rank))."""
    raise NotImplementedError


def retrieve(query: str, top_k: int = 5) -> list[tuple[KnowledgeUnit, float]]:
    """Public entrypoint: hybrid retrieve + relevance gate (drop weak matches)."""
    raise NotImplementedError
