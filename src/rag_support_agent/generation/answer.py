"""Answer generation with grounding, citations, and (grounding-only) abstention.

The agent answers ONLY from retrieved context — never from parametric memory — attaches
citations, and abstains rather than guess. One confident hallucination costs more trust
than ten honest "I don't know"s.

Grounding is enforced in two independent layers:

  1. Structural (provider-independent, un-foolable): if the M2 relevance gate returns
     nothing, the query is out of scope, so we never even call a generator — we abstain.
     This is the M2 gate closing the loop into a refusal.
  2. Synthesis: the generator itself is grounded — ExtractiveGenerator by construction
     (verbatim echo), GeminiGenerator by a strict prompt that can emit a refusal sentinel.

``build_answer`` is the pure seam: hand it fabricated ``RetrievalResult``s and a generator
and it produces an ``Answer`` with no database in the loop (see ``tests/test_generation``).
``answer_question`` is the thin DB-backed wrapper that supplies real retrieval.

M4 boundary: abstention here is *grounding-only* (empty retrieval or generator refusal).
Calibrated, confidence-threshold abstention is M4 — so ``compute_confidence`` stays a stub
and ``Answer.confidence`` is a documented placeholder that nothing yet acts on.
"""

from __future__ import annotations

import time

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.generation.generators import Generator, get_generator, parse_citations
from rag_support_agent.knowledge.models import (
    Answer,
    AnswerVerdict,
    Citation,
    KnowledgeUnit,
    RetrievalResult,
)
from rag_support_agent.retrieval.hybrid import retrieve

# Shown when the relevance gate returned nothing — the honest out-of-scope refusal.
NO_SOURCE_MESSAGE = (
    "I don't have a source for that in the knowledge base, so I won't guess. "
    "(No retrieved passage cleared the relevance gate.)"
)


def compute_confidence(query: str, retrieved: list[tuple[KnowledgeUnit, float]]) -> float:
    """Confidence in [0,1] from retrieval score spread + grounding check.

    Signals to combine (M4):
      - top score and gap to the next result (a clear winner => higher confidence)
      - whether the drafted answer is entailed by the retrieved context
      - optional lightweight self-eval
    """
    raise NotImplementedError


def _placeholder_confidence(results: list[RetrievalResult]) -> float:
    """M3 placeholder: the top result's fused RRF score.

    Deliberately trivial and *not* consumed by any decision in M3 — abstention here keys
    off retrieval/grounding, never off this number. M4 replaces it with a calibrated signal
    (score spread + grounding) via ``compute_confidence``.
    """
    return results[0].score if results else 0.0


def _abstain(text: str, latency_ms: float | None = None) -> Answer:
    return Answer(
        verdict=AnswerVerdict.ABSTAINED,
        text=text,
        confidence=0.0,
        citations=[],
        latency_ms=latency_ms,
    )


def build_answer(
    query: str, results: list[RetrievalResult], generator: Generator
) -> Answer:
    """Turn retrieved results + a generator into a grounded, cited ``Answer``. Pure (no DB).

    Layer 1 — structural grounding: empty ``results`` means the gate found nothing in
    scope, so we abstain without calling the generator (no chance to invent an answer).
    Otherwise we synthesize, then build the citation list from the ``[n]`` markers the
    generator actually used — so a citation always points at a passage that was really cited.
    """
    if not results:
        return _abstain(NO_SOURCE_MESSAGE)

    started = time.perf_counter()
    gen = generator.generate(query, results)
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Layer 2 — synthesis grounding: the generator refused (context insufficient).
    if gen.abstained:
        return _abstain(gen.text, latency_ms)

    # Ascending marker order for the citation list (footnote convention); each Citation
    # keeps its true marker so [4] in the prose renders as [4] in the list, not [2].
    cited = sorted(parse_citations(gen.text, n=len(results)))
    citations = [
        Citation(
            index=i,
            knowledge_unit_id=results[i - 1].unit.id,
            source_uri=results[i - 1].unit.source_uri,
            score=results[i - 1].score,
        )
        for i in cited
    ]
    return Answer(
        verdict=AnswerVerdict.ANSWERED,
        text=gen.text,
        confidence=_placeholder_confidence(results),
        citations=citations,
        latency_ms=latency_ms,
    )


def answer_question(
    query: str,
    top_k: int | None = None,
    generator: Generator | None = None,
    settings: Settings | None = None,
) -> Answer:
    """End-to-end: retrieve (hybrid + gate) -> ground -> cite -> answer or abstain.

    The only DB-touching step is ``retrieve``; everything after it is the pure
    ``build_answer``. ``cost_usd`` and stale-source flags are left unset here — cost
    accounting is M7 and freshness is M6.
    """
    s = settings or get_settings()
    results = retrieve(query, top_k=top_k, settings=s)
    gen = generator or get_generator(s)
    return build_answer(query, results, gen)
