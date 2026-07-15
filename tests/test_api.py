"""Tests for the M8 API + thin chat UI.

All DB-free and key-free, like the rest of the suite:

  - ``serialize_answer`` / ``sse_event`` are pure — tested on a fabricated ``Answer``.
  - ``stream_answer`` is pinned to ``build_answer`` on the *same* fabricated inputs, so the
    streaming seam and the blocking seam cannot diverge (the M5 runner-vs-build_answer move,
    reused here).
  - ``/health``, ``/ask`` and ``/ask/stream`` are driven with FastAPI's ``TestClient`` while
    ``answer_question`` / ``answer_question_stream`` are monkeypatched to canned values, so no
    Postgres and no API key are ever touched.
"""

from fastapi.testclient import TestClient

from rag_support_agent.api.serialize import serialize_answer, sse_event
from rag_support_agent.api.server import create_app
from rag_support_agent.generation.answer import build_answer, stream_answer
from rag_support_agent.generation.generators import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SENTINEL,
    ExtractiveGenerator,
    GenResult,
)
from rag_support_agent.knowledge.models import (
    Answer,
    AnswerVerdict,
    Citation,
    KnowledgeUnit,
    RetrievalResult,
)


def _result(uid, content, score, dense=None):
    unit = KnowledgeUnit(id=uid, content=content, source_uri=f"{uid}.md", section=uid)
    return RetrievalResult(unit=unit, score=score, dense_similarity=dense)


def _canned_answer():
    return Answer(
        verdict=AnswerVerdict.ANSWERED,
        text="Go to Settings and click Rotate [1].",
        confidence=0.34,
        citations=[Citation(index=1, knowledge_unit_id="u1", source_uri="api-keys.md", score=0.0328)],
        stale_sources=["api-keys.md"],
        latency_ms=12.3,
        cost_usd=0.0,
    )


# --- serialize_answer (pure) --------------------------------------------------------


def test_serialize_answer_carries_every_trust_signal():
    payload = serialize_answer(_canned_answer(), "How do I rotate a key?", "extractive")
    assert payload["query"] == "How do I rotate a key?"
    assert payload["provider"] == "extractive"
    assert payload["verdict"] == "answered"
    assert payload["confidence"] == 0.34
    assert payload["stale_sources"] == ["api-keys.md"]
    assert payload["cost_usd"] == 0.0
    assert payload["latency_ms"] == 12.3
    # Citation keeps its true marker index and source, not a re-enumeration.
    assert payload["citations"] == [
        {"index": 1, "knowledge_unit_id": "u1", "source_uri": "api-keys.md", "score": 0.0328}
    ]


def test_serialize_answer_abstained_has_no_citations():
    a = Answer(verdict=AnswerVerdict.ABSTAINED, text="I don't have a source.", confidence=0.0)
    payload = serialize_answer(a, "bake bread", "extractive")
    assert payload["verdict"] == "abstained"
    assert payload["citations"] == []
    assert payload["stale_sources"] == []


# --- sse_event (pure) ---------------------------------------------------------------


def test_sse_event_frames_a_named_json_record():
    frame = sse_event("token", {"text": "hello"})
    assert frame == 'event: token\ndata: {"text": "hello"}\n\n'


def test_sse_event_keeps_non_ascii():
    frame = sse_event("token", {"text": "café —"})
    assert "café —" in frame  # ensure_ascii=False, so the UI gets the real characters


# --- stream_answer is pinned to build_answer (pure, DB-free) -------------------------


class _FakeStreamGen:
    """generate() returns the concatenation of what stream() yields — so the two seams get
    byte-identical text and any divergence in the *logic* around it shows up as a failure."""

    model = "fake"

    def __init__(self, chunks, abstained=False):
        self._chunks = chunks
        self._abstained = abstained

    def generate(self, query, results):
        return GenResult("".join(self._chunks), abstained=self._abstained)

    def stream(self, query, results):
        yield from self._chunks


def _collect(stream):
    tokens, answer = [], None
    for item in stream:
        if isinstance(item, Answer):
            answer = item
        else:
            tokens.append(item)
    return tokens, answer


def _key(a):
    """Comparable identity of an Answer, minus the (timing-dependent) latency."""
    return (
        a.verdict,
        a.text,
        a.confidence,
        [(c.index, c.knowledge_unit_id, c.source_uri, c.score) for c in a.citations],
        list(a.stale_sources),
    )


def test_stream_matches_build_answered():
    # Clear-winner dense spread so both clear the M4 confidence threshold and answer.
    results = [_result("a", "alpha", 0.030, dense=0.80), _result("b", "bravo", 0.020, dense=0.20)]
    blocking = build_answer("q", results, ExtractiveGenerator())
    _, streamed = _collect(stream_answer("q", results, ExtractiveGenerator()))
    assert streamed.verdict is AnswerVerdict.ANSWERED
    assert _key(streamed) == _key(blocking)


def test_stream_matches_build_low_confidence_abstain():
    # Flat dense field -> no clear winner -> Layer 3 abstain, in *both* seams.
    results = [
        _result("billing", "x", 0.031, dense=0.44),
        _result("webhooks", "y", 0.030, dense=0.43),
        _result("errors", "z", 0.029, dense=0.43),
    ]
    blocking = build_answer("q", results, ExtractiveGenerator())
    _, streamed = _collect(stream_answer("q", results, ExtractiveGenerator()))
    assert streamed.verdict is AnswerVerdict.ABSTAINED
    assert _key(streamed) == _key(blocking)


def test_stream_matches_build_sentinel_abstain():
    results = [_result("a", "alpha", 0.03, dense=0.80), _result("b", "bravo", 0.02, dense=0.20)]
    gen_blocking = _FakeStreamGen([INSUFFICIENT_CONTEXT_MESSAGE], abstained=True)
    gen_stream = _FakeStreamGen([SENTINEL])  # the raw sentinel the model would emit
    blocking = build_answer("q", results, gen_blocking)
    _, streamed = _collect(stream_answer("q", results, gen_stream))
    assert streamed.verdict is AnswerVerdict.ABSTAINED
    assert streamed.text == INSUFFICIENT_CONTEXT_MESSAGE
    assert _key(streamed) == _key(blocking)


def test_stream_empty_retrieval_abstains_without_touching_generator():
    class _Boom:
        def generate(self, query, results):
            raise AssertionError("generator must not run on empty retrieval")

        def stream(self, query, results):
            raise AssertionError("generator must not run on empty retrieval")

    _, streamed = _collect(stream_answer("q", [], _Boom()))
    assert streamed.verdict is AnswerVerdict.ABSTAINED
    assert streamed.citations == []


def test_stream_emits_answer_text_as_tokens_then_final():
    results = [_result("a", "alpha", 0.030, dense=0.80), _result("b", "bravo", 0.020, dense=0.20)]
    gen = _FakeStreamGen(["To rotate a key, click ", "Rotate. ", "[1]"])
    tokens, streamed = _collect(stream_answer("q", results, gen))
    assert "".join(tokens) == "To rotate a key, click Rotate. [1]"
    assert streamed.text == "To rotate a key, click Rotate. [1]"
    assert [c.index for c in streamed.citations] == [1]


# --- endpoints (TestClient, monkeypatched pipeline) ---------------------------------


def test_health():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index_serves_the_ui():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "RAG Support Agent" in r.text
    assert "/ask/stream" in r.text  # the UI wires itself to the SSE endpoint


def test_ask_returns_serialized_answer(monkeypatch):
    def fake(query, top_k=None, settings=None, record_event=False):
        assert record_event is True  # live UI traffic must feed the M7 blind-spot log
        return _canned_answer()

    monkeypatch.setattr("rag_support_agent.generation.answer.answer_question", fake)
    client = TestClient(create_app())
    r = client.post("/ask", json={"query": "How do I rotate an API key?"})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "answered"
    assert body["provider"] == "extractive"
    assert body["citations"][0]["source_uri"] == "api-keys.md"


def test_ask_rejects_empty_query():
    client = TestClient(create_app())
    r = client.post("/ask", json={"query": "   "})
    assert r.status_code == 422


def test_ask_stream_emits_token_then_done(monkeypatch):
    def fake(query, top_k=None, settings=None, record_event=False):
        assert record_event is True
        yield ("token", "Rotate ")
        yield ("token", "via [1].")
        yield ("answer", _canned_answer())

    monkeypatch.setattr("rag_support_agent.generation.answer.answer_question_stream", fake)
    client = TestClient(create_app())
    r = client.get("/ask/stream", params={"q": "How do I rotate an API key?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "event: token" in body
    assert "event: done" in body
    assert '"verdict": "answered"' in body
    assert "api-keys.md" in body


def test_ask_stream_rejects_empty_query():
    client = TestClient(create_app())
    r = client.get("/ask/stream", params={"q": "  "})
    assert r.status_code == 422
