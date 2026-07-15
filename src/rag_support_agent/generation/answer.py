"""Answer generation with grounding, citations, and confidence-based abstention.

The agent answers ONLY from retrieved context — never from parametric memory — attaches
citations, and abstains rather than guess. One confident hallucination costs more trust
than ten honest "I don't know"s.

Grounding is enforced in two independent layers (M3):

  1. Structural (provider-independent, un-foolable): if the M2 relevance gate returns
     nothing, the query is out of scope, so we never even call a generator — we abstain.
     This is the M2 gate closing the loop into a refusal.
  2. Synthesis: the generator itself is grounded — ExtractiveGenerator by construction
     (verbatim echo), GeminiGenerator by a strict prompt that can emit a refusal sentinel.

``build_answer`` is the pure seam: hand it fabricated ``RetrievalResult``s and a generator
and it produces an ``Answer`` with no database in the loop (see ``tests/test_generation``).
``answer_question`` is the thin DB-backed wrapper that supplies real retrieval.

Abstention now has *three* independent layers (Layers 1–2 are M3 grounding, Layer 3 is M4):

  3. Confidence (M4): retrieval brought something back AND the generator answered, but the
     top hit doesn't stand out from the field — an ambiguous, no-clear-winner retrieval. We
     compute a confidence *spread* signal and abstain below ``confidence_abstain_threshold``,
     pointing at the closest source rather than the "nothing here" of the structural refusal.

M4/M5 boundary: M4 computes the signal, wires *one* defensible threshold, and adds the third
abstention. Calibrating that threshold against the labeled eval set (abstention
precision/recall) — and the optional LLM self-eval grounding factor — is M5.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.eval.cost import TokenUsage, estimate_cost_usd
from rag_support_agent.generation.generators import Generator, get_generator, parse_citations
from rag_support_agent.knowledge.freshness import assess_freshness
from rag_support_agent.knowledge.models import (
    Answer,
    AnswerVerdict,
    Citation,
    RetrievalResult,
)
from rag_support_agent.retrieval.hybrid import retrieve

logger = logging.getLogger(__name__)

# Shown when the relevance gate returned nothing — the honest out-of-scope refusal.
NO_SOURCE_MESSAGE = (
    "I don't have a source for that in the knowledge base, so I won't guess. "
    "(No retrieved passage cleared the relevance gate.)"
)


def _retrieval_spread(results: list[RetrievalResult]) -> float:
    """Separation of the top hit from the field, on dense cosine similarity, in [0,1].

    ``spread = (d_top - mean(dense of the other results)) / d_top`` (clamped to [0,1]).
    A clear winner — top cosine well above the pack — scores high; a flat field of
    near-ties (ambiguous *or* out-of-scope) scores ~0. It is deliberately a *relative*
    signal: the fused RRF score is rank-based and magnitude-blind (it reads ~0.033 vs
    ~0.032 even for an obvious answer), and an absolute cosine floor doesn't transfer
    across embedders (M2), whereas this top-vs-field gap does.

    Keyed off the *fused winner* (``results[0]``) — the passage we would actually serve —
    not the max-dense passage in the set. Needs ≥2 results carrying a dense similarity: a
    lone gated hit has no field to stand out from, so a winner cannot be established and we
    return 0.0 (abstain conservatively) rather than claim false confidence.
    """
    if len(results) < 2:
        return 0.0
    d_top = results[0].dense_similarity
    if d_top is None or d_top <= 0:
        return 0.0
    field = [r.dense_similarity for r in results[1:] if r.dense_similarity is not None]
    if not field:
        return 0.0
    spread = (d_top - sum(field) / len(field)) / d_top
    return max(0.0, min(1.0, spread))


def _grounding_factor() -> float:
    """Grounding component of confidence, in [0,1].

    Spread answers "is there a clear winner"; grounding answers "is the drafted answer
    actually supported by that winner". For *both* shipped generators this is 1.0 by
    design: ``ExtractiveGenerator`` echoes retrieved text verbatim (grounded by
    construction), and ``GeminiGenerator`` is constrained to the numbered context and
    emits the sentinel otherwise (caught as Layer-2 abstention before we get here). The
    optional M4 enhancement — an LLM self-eval scoring entailment of the draft by its
    cited context — plugs in *here* as a <1.0 factor for the gemini path; it stays off by
    default so the pipeline needs no API key. Measured faithfulness becomes a number in M5.
    """
    return 1.0


def compute_confidence(query: str, results: list[RetrievalResult]) -> float:
    """Confidence in [0,1] that the retrieved context yields a trustworthy answer.

    ``confidence = retrieval_spread × grounding_factor``. The backbone is the *spread* of
    the top hit over the field (see ``_retrieval_spread``); ``grounding_factor`` is 1.0 for
    the shipped generators (see ``_grounding_factor``). ``query`` is unused today — it is
    the hook the optional gemini self-eval would key off (query + draft → entailment).
    """
    return _retrieval_spread(results) * _grounding_factor()


def _closest_source(result: RetrievalResult) -> str:
    label = result.unit.source_uri
    if result.unit.section:
        label = f"{label} :: {result.unit.section}"
    return label


def _low_confidence_message(results: list[RetrievalResult]) -> str:
    """Layer-3 refusal: unlike the structural "nothing here", point at the closest source."""
    return (
        "I don't have a confident answer for that — the retrieved passages are too "
        "ambiguous, with no clear best match. The closest source is "
        f"{_closest_source(results[0])}; check it directly or try rephrasing the question."
    )


def _abstain(
    text: str, latency_ms: float | None = None, confidence: float = 0.0
) -> Answer:
    return Answer(
        verdict=AnswerVerdict.ABSTAINED,
        text=text,
        confidence=confidence,
        citations=[],
        latency_ms=latency_ms,
    )


def _stale_cited_sources(
    results: list[RetrievalResult],
    citations: list[Citation],
    settings: Settings,
    now: datetime | None,
) -> list[str]:
    """Freshness flag (M6): possibly-stale sources *among the ones backing this answer*.

    Freshness is assessed over the full retrieved set (that field is what the relative
    age-outlier signal needs), but we surface only the sources the reader actually sees —
    the cited ones — so the warning matches the answer's footnotes. If the generator cited
    nothing, we fall back to the served top passage's source.
    """
    report = assess_freshness(results, settings=settings, now=now)
    if not report.stale_sources:
        return []
    shown = {c.source_uri for c in citations} or {results[0].unit.source_uri}
    return [src for src in report.stale_sources if src in shown]


def build_answer(
    query: str,
    results: list[RetrievalResult],
    generator: Generator,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> Answer:
    """Turn retrieved results + a generator into a grounded, cited ``Answer``. Pure (no DB).

    Layer 1 — structural grounding: empty ``results`` means the gate found nothing in
    scope, so we abstain without calling the generator (no chance to invent an answer).
    Otherwise we synthesize; Layer 2 catches a generator refusal. Then Layer 3 (M4): even a
    fully-formed answer is withheld if confidence (retrieval spread) is below threshold —
    an ambiguous retrieval with no clear winner. Only past all three do we build the
    citation list from the ``[n]`` markers the generator actually used, so a citation always
    points at a passage that was really cited. Finally we attach the M6 freshness flag over
    the cited sources (``now`` is injectable so the check is deterministic in tests).
    """
    s = settings or get_settings()
    if not results:
        return _abstain(NO_SOURCE_MESSAGE)

    started = time.perf_counter()
    gen = generator.generate(query, results)
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Layer 2 — synthesis grounding: the generator refused (context insufficient).
    if gen.abstained:
        return _abstain(gen.text, latency_ms)

    # Layer 3 — confidence (M4): the answer exists but the retrieval was ambiguous. Abstain
    # below threshold, surfacing the actual (low) confidence and the closest source.
    confidence = compute_confidence(query, results)
    if confidence < s.confidence_abstain_threshold:
        return _abstain(_low_confidence_message(results), latency_ms, confidence)

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
        confidence=confidence,
        citations=citations,
        stale_sources=_stale_cited_sources(results, citations, s, now),
        latency_ms=latency_ms,
    )


def _generation_cost(usage: TokenUsage, generator: Generator) -> float:
    """Per-request serving cost in USD from this query's generation token usage (M7).

    Cross-provider: prices ``usage`` at whichever model actually ran (``generator.model``),
    reusing ``eval/cost.py``'s table — whose scope note reserves exactly this wiring for M7.
    The keyless ``ExtractiveGenerator`` spends no tokens → $0, so the default path stays free.
    """
    if usage.input_tokens == 0 and usage.output_tokens == 0:
        return 0.0
    return estimate_cost_usd(usage, getattr(generator, "model", None))


def _record_event(
    query: str,
    answer: Answer,
    results: list[RetrievalResult],
    usage: TokenUsage,
    latency_ms: float,
    settings: Settings,
) -> None:
    """Append this served query to the blind-spot / observability log (M7).

    Imported lazily so the pure generation path never pulls in psycopg at import time. The
    logged ``latency_ms`` is the *end-to-end* request time (retrieve → answer, what the user
    waits) — deliberately wider than ``Answer.latency_ms``, which M6 scoped to the generation
    slice alone. ``top_source`` is the closest served passage (``None`` when the gate returned
    nothing).
    """
    from rag_support_agent.observability.blindspot import QueryEvent, record

    record(
        QueryEvent(
            query=query,
            confidence=answer.confidence,
            abstained=answer.verdict is AnswerVerdict.ABSTAINED,
            top_source=results[0].unit.source_uri if results else None,
            latency_ms=latency_ms,
            cost_usd=answer.cost_usd or 0.0,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ),
        settings=settings,
    )


def answer_question(
    query: str,
    top_k: int | None = None,
    generator: Generator | None = None,
    settings: Settings | None = None,
    record_event: bool = False,
) -> Answer:
    """End-to-end: retrieve (hybrid + gate) -> ground -> cite -> answer or abstain.

    The only DB-touching step is ``retrieve``; everything after it is the pure
    ``build_answer`` — which also attaches the M6 freshness flag. On top of that pure core M7
    adds two thin layers: it prices this query's generation tokens onto ``Answer.cost_usd``
    (cross-provider; $0 keyless), and — only when ``record_event`` is set — appends a
    ``QueryEvent`` to the blind-spot log. Logging is opt-in (eval/tests call ``build_answer``
    directly and never touch it) and best-effort: a telemetry failure is logged and
    swallowed, never failing the answer.
    """
    s = settings or get_settings()
    started = time.perf_counter()
    results = retrieve(query, top_k=top_k, settings=s)
    gen = generator or get_generator(s)
    # Reset per-call token counters so cost reflects *this* query even on a reused generator
    # (a Layer-1 abstain never calls generate(), so stale counts would otherwise leak in).
    for attr in ("last_input_tokens", "last_output_tokens"):
        if hasattr(gen, attr):
            setattr(gen, attr, 0)

    answer = build_answer(query, results, gen, settings=s)
    usage = TokenUsage(
        input_tokens=getattr(gen, "last_input_tokens", 0),
        output_tokens=getattr(gen, "last_output_tokens", 0),
    )
    answer.cost_usd = _generation_cost(usage, gen)
    latency_ms = (time.perf_counter() - started) * 1000.0  # end-to-end, for the log

    if record_event:
        try:
            _record_event(query, answer, results, usage, latency_ms, s)
        except Exception as exc:  # telemetry must never fail a user's answer
            logger.warning("query-event logging failed: %s", exc)
    return answer
