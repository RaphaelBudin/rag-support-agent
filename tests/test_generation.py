"""Unit tests for generation: grounding, citation extraction, and abstention.

All DB-free and key-free — they exercise the pure ``build_answer`` seam with fabricated
retrieval results plus a fake or extractive generator, so the grounding *contract* is
pinned down without Postgres or an API in the loop.
"""

from rag_support_agent.generation.answer import NO_SOURCE_MESSAGE, build_answer
from rag_support_agent.generation.generators import (
    ExtractiveGenerator,
    GenResult,
    parse_citations,
)
from rag_support_agent.knowledge.models import AnswerVerdict, KnowledgeUnit, RetrievalResult


def _result(uid: str, content: str, score: float) -> RetrievalResult:
    unit = KnowledgeUnit(id=uid, content=content, source_uri=f"{uid}.md", section=uid)
    return RetrievalResult(unit=unit, score=score)


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
    results = [
        _result("a", "alpha", 0.030),
        _result("b", "bravo", 0.020),
        _result("c", "charlie", 0.010),
    ]
    # The generator's answer cites only passage [2] -> only unit "b" becomes a citation.
    answer = build_answer("q", results, _FakeGenerator(GenResult(text="per [2], do X")))
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert [c.knowledge_unit_id for c in answer.citations] == ["b"]
    assert answer.citations[0].index == 2  # the true marker, not a re-enumeration to [1]
    assert answer.citations[0].source_uri == "b.md"
    assert answer.citations[0].score == 0.020
    # Placeholder confidence is the top result's fused score (M4 replaces this).
    assert answer.confidence == 0.030


def test_citations_keep_sparse_markers_aligned():
    # A model that cites [1] and [3] (skipping [2]) must yield citations numbered [1],[3] —
    # the inline markers stay stable handles into the retrieved set.
    results = [_result("a", "alpha", 0.03), _result("b", "bravo", 0.02), _result("c", "charlie", 0.01)]
    answer = build_answer("q", results, _FakeGenerator(GenResult(text="[3] says X, and [1] too")))
    assert [(c.index, c.knowledge_unit_id) for c in answer.citations] == [(1, "a"), (3, "c")]


# --- extractive generator: grounded by construction ---------------------------------


def test_extractive_is_grounded_and_cited():
    results = [
        _result("a", "To rotate a key, click Rotate.", 0.030),
        _result("b", "Revoking disables a key at once.", 0.020),
    ]
    answer = build_answer("how do I rotate a key?", results, ExtractiveGenerator())
    assert answer.verdict is AnswerVerdict.ANSWERED
    # Every character of the answer comes from retrieved content (no invention).
    assert "To rotate a key, click Rotate." in answer.text
    # Both echoed passages are cited, in order.
    assert [c.knowledge_unit_id for c in answer.citations] == ["a", "b"]
