"""Core domain models.

A *knowledge unit* is the atom of the system: a retrievable piece of knowledge
plus the metadata needed to reason about its freshness and provenance. Keeping
freshness on the unit (not just at query time) is what makes decay detection and
blind-spot reporting possible later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class KnowledgeUnit:
    """A single retrievable unit of knowledge."""

    id: str
    content: str
    source_uri: str            # where it came from (doc URL, ticket id, ...)
    section: str | None = None
    chunk_index: int = 0       # position within its source (part of the deterministic id)
    content_hash: str = ""     # dedup + change detection across re-ingests
    created_at: datetime | None = None
    source_updated_at: datetime | None = None  # drives freshness/decay (see knowledge.freshness)
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """One retrieved unit with its ranking signal and the raw component scores.

    ``score`` is the fused Reciprocal Rank Fusion score (the ranking signal). The
    component fields are kept alongside it on purpose: RRF throws away magnitude, but
    the relevance gate (here) and the confidence layer (M4) still need the absolute,
    interpretable dense similarity and the raw BM25 score. ``None`` means that arm did
    not return this unit in its candidate pool.
    """

    unit: KnowledgeUnit
    score: float                          # fused RRF score
    dense_similarity: float | None = None  # cosine similarity, ~[0,1], absolute
    sparse_score: float | None = None      # BM25 score (corpus-dependent scale)
    dense_rank: int | None = None          # 1-based rank in the dense arm
    sparse_rank: int | None = None         # 1-based rank in the sparse arm


class AnswerVerdict(str, Enum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"    # confidence below threshold — see generation.answer


@dataclass
class Citation:
    knowledge_unit_id: str
    source_uri: str
    score: float


@dataclass
class Answer:
    """The result returned to the user, with the trust signals attached."""

    verdict: AnswerVerdict
    text: str
    confidence: float                       # 0..1
    citations: list[Citation] = field(default_factory=list)
    stale_sources: list[str] = field(default_factory=list)  # flagged by freshness check
    latency_ms: float | None = None
    cost_usd: float | None = None
