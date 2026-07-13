"""Unit tests for generation: grounding, citation extraction, and abstention.

All DB-free and key-free — they exercise the pure ``build_answer`` seam with fabricated
retrieval results plus a fake or extractive generator, so the grounding *contract* is
pinned down without Postgres or an API in the loop.
"""

from rag_support_agent.generation.answer import (
    NO_SOURCE_MESSAGE,
    build_answer,
    compute_confidence,
)
from rag_support_agent.generation.generators import (
    ExtractiveGenerator,
    GenResult,
    parse_citations,
)
from rag_support_agent.knowledge.models import AnswerVerdict, KnowledgeUnit, RetrievalResult


def _result(
    uid: str, content: str, score: float, dense: float | None = None
) -> RetrievalResult:
    unit = KnowledgeUnit(id=uid, content=content, source_uri=f"{uid}.md", section=uid)
    return RetrievalResult(unit=unit, score=score, dense_similarity=dense)


class _FakeGenerator:
    """A generator with a scripted output — lets us test assembly independent of any LLM."""

    def __init__(self, result: GenResult) -> None:
        self._result = result

    def generate(self, query, results) -> GenResult:
        return self._result


# --- parse_citations (pure) ---------------------------------------------------------


def test_parse_citations_dedupes_and_preserves_order():
    assert parse_citations("uses [2] then [1] then [2] again", n=3) == [2, 1]


def test_parse_citations_filters_out_of_range_markers():
    # A [9] when only 3 passages were supplied is a hallucinated handle -> dropped.
    assert parse_citations("see [1] and [9]", n=3) == [1]


def test_parse_citations_none_present():
    assert parse_citations("no markers at all", n=3) == []


# --- structural grounding: empty retrieval -> abstain -------------------------------


def test_empty_retrieval_abstains_without_calling_generator():
    class _Boom:
        def generate(self, query, results):
            raise AssertionError("generator must not be called when nothing was retrieved")

    answer = build_answer("anything", [], _Boom())
    assert answer.verdict is AnswerVerdict.ABSTAINED
    assert answer.text == NO_SOURCE_MESSAGE
    assert answer.citations == []
    assert answer.confidence == 0.0


# --- synthesis grounding: generator refusal -> abstain ------------------------------


def test_generator_refusal_abstains():
    results = [_result("a", "some content", 0.03)]
    answer = build_answer("q", results, _FakeGenerator(GenResult(text="nope", abstained=True)))
    assert answer.verdict is AnswerVerdict.ABSTAINED
    assert answer.citations == []


# --- citations reflect only the markers actually used -------------------------------


def test_citations_map_to_cited_passages_only():
    # Clear-winner dense spread so the answer clears the M4 confidence threshold.
    results = [
        _result("a", "alpha", 0.030, dense=0.80),
        _result("b", "bravo", 0.020, dense=0.30),
        _result("c", "charlie", 0.010, dense=0.20),
    ]
    # The generator's answer cites only passage [2] -> only unit "b" becomes a citation.
    answer = build_answer("q", results, _FakeGenerator(GenResult(text="per [2], do X")))
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert [c.knowledge_unit_id for c in answer.citations] == ["b"]
    assert answer.citations[0].index == 2  # the true marker, not a re-enumeration to [1]
    assert answer.citations[0].source_uri == "b.md"
    assert answer.citations[0].score == 0.020  # citation carries the fused RRF score
    # Confidence is now the retrieval spread (M4), no longer the RRF placeholder.
    assert answer.confidence == compute_confidence("q", results)
    assert answer.confidence > 0.12


def test_citations_keep_sparse_markers_aligned():
    # A model that cites [1] and [3] (skipping [2]) must yield citations numbered [1],[3] —
    # the inline markers stay stable handles into the retrieved set.
    results = [
        _result("a", "alpha", 0.03, dense=0.80),
        _result("b", "bravo", 0.02, dense=0.30),
        _result("c", "charlie", 0.01, dense=0.20),
    ]
    answer = build_answer("q", results, _FakeGenerator(GenResult(text="[3] says X, and [1] too")))
    assert [(c.index, c.knowledge_unit_id) for c in answer.citations] == [(1, "a"), (3, "c")]


# --- extractive generator: grounded by construction ---------------------------------


def test_extractive_is_grounded_and_cited():
    results = [
        _result("a", "To rotate a key, click Rotate.", 0.030, dense=0.80),
        _result("b", "Revoking disables a key at once.", 0.020, dense=0.20),
    ]
    answer = build_answer("how do I rotate a key?", results, ExtractiveGenerator())
    assert answer.verdict is AnswerVerdict.ANSWERED
    # Every character of the answer comes from retrieved content (no invention).
    assert "To rotate a key, click Rotate." in answer.text
    # Both echoed passages are cited, in order.
    assert [c.knowledge_unit_id for c in answer.citations] == ["a", "b"]


# --- confidence-based abstention (M4): ambiguous retrieval -> abstain (Layer 3) ------


def test_low_confidence_abstains_pointing_at_closest_source():
    # A flat field of near-tie dense similarities => no clear winner => low spread. The
    # generator answers (extractive can't refuse), but Layer 3 withholds it.
    results = [
        _result("billing", "some passage", 0.031, dense=0.44),
        _result("webhooks", "another", 0.030, dense=0.43),
        _result("errors", "yet another", 0.029, dense=0.43),
    ]
    answer = build_answer("what should I do", results, ExtractiveGenerator())
    assert answer.verdict is AnswerVerdict.ABSTAINED
    assert answer.confidence < 0.12
    assert answer.citations == []
    # Distinct from the structural "nothing here": Layer 3 names the closest source.
    assert answer.text != NO_SOURCE_MESSAGE
    assert "billing.md :: billing" in answer.text


def test_high_spread_answers_with_that_confidence():
    results = [
        _result("a", "the answer", 0.031, dense=0.70),
        _result("b", "off-topic", 0.030, dense=0.20),
        _result("c", "off-topic", 0.029, dense=0.18),
    ]
    answer = build_answer("q", results, ExtractiveGenerator())
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert answer.confidence == compute_confidence("q", results)
    assert answer.confidence >= 0.12


def test_single_gated_result_abstains_no_field_to_beat():
    # One lone hit (even a dense-strong one) has no field to stand out from -> spread 0.
    results = [_result("a", "lonely passage", 0.031, dense=0.9)]
    answer = build_answer("q", results, ExtractiveGenerator())
    assert answer.verdict is AnswerVerdict.ABSTAINED
    assert answer.confidence == 0.0


# --- compute_confidence / retrieval spread (pure) -----------------------------------


def test_spread_high_for_clear_winner():
    results = [_result("a", "x", 0.0, dense=0.60), _result("b", "y", 0.0, dense=0.20)]
    assert compute_confidence("q", results) > 0.5


def test_spread_zero_for_flat_field():
    results = [_result("a", "x", 0.0, dense=0.44), _result("b", "y", 0.0, dense=0.44)]
    assert compute_confidence("q", results) == 0.0


def test_spread_zero_when_winner_lacks_dense():
    # Winner came only from the sparse arm (no dense sim) -> can't measure spread -> 0.
    results = [_result("a", "x", 0.0, dense=None), _result("b", "y", 0.0, dense=0.30)]
    assert compute_confidence("q", results) == 0.0
