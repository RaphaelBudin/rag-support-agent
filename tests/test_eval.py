"""Unit tests for the evaluation harness — DB-free and key-free.

Two things get pinned here: (1) the pure metric functions compute what they claim (an eval
harness whose metrics are wrong is worse than none), and (2) the instrumented runner's verdict
stays identical to ``generation.answer.build_answer`` — so the numbers the harness reports are
the numbers the real pipeline would produce, not a parallel re-implementation that can drift.
"""

from rag_support_agent.config import Settings
from rag_support_agent.eval.faithfulness import _parse_verdicts
from rag_support_agent.eval.metrics import (
    ConfusionMatrix,
    abstention_confusion,
    aggregate_faithfulness,
    normalize_source,
    percentile,
    recall_at_k,
)
from rag_support_agent.eval.run import Reason, abstains_at, evaluate_from_results, load_dataset
from rag_support_agent.generation.answer import build_answer
from rag_support_agent.generation.generators import ExtractiveGenerator, GenResult
from rag_support_agent.knowledge.models import AnswerVerdict, KnowledgeUnit, RetrievalResult

_SETTINGS = Settings(confidence_abstain_threshold=0.12)


def _result(uid: str, content: str, score: float, dense: float | None = None) -> RetrievalResult:
    unit = KnowledgeUnit(id=uid, content=content, source_uri=f"{uid}.md", section=uid)
    return RetrievalResult(unit=unit, score=score, dense_similarity=dense)


class _FakeGenerator:
    def __init__(self, result: GenResult) -> None:
        self._result = result

    def generate(self, query, results) -> GenResult:
        return self._result


# --- recall_at_k --------------------------------------------------------------------


def test_recall_hit_within_k():
    assert recall_at_k("api-keys.md", ["errors.md", "api-keys.md", "x.md"], k=5) is True


def test_recall_miss_outside_k():
    # gold sits at index 1, but k=1 only looks at the first result.
    assert recall_at_k("api-keys.md", ["errors.md", "api-keys.md"], k=1) is False


def test_recall_list_gold_matches_any():
    assert recall_at_k(["a.md", "b.md"], ["z.md", "b.md"], k=5) is True


def test_recall_none_gold_is_excluded():
    # No gold source (out-of-scope / ambiguous) -> tri-state None, not a 0.
    assert recall_at_k(None, ["a.md"], k=5) is None


def test_recall_normalizes_paths():
    assert normalize_source("sample_docs/api-keys.md") == "api-keys.md"
    assert recall_at_k("sample_docs/api-keys.md", ["api-keys.md"], k=5) is True


# --- abstention confusion -----------------------------------------------------------


def test_abstention_confusion_and_rates():
    m = abstention_confusion([(True, True), (True, False), (False, True), (False, False)])
    assert (m.tp, m.fp, m.fn, m.tn) == (1, 1, 1, 1)
    assert m.precision == 0.5
    assert m.recall == 0.5
    assert m.f1 == 0.5


def test_abstention_precision_none_when_never_abstains():
    m = abstention_confusion([(False, False), (False, True)])
    assert m.precision is None       # never abstained -> precision undefined
    assert m.recall == 0.0           # one should-abstain missed


def test_abstention_recall_none_when_none_should():
    m = abstention_confusion([(True, False), (False, False)])
    assert m.recall is None          # nothing should abstain -> recall undefined


def test_confusion_perfect():
    m = ConfusionMatrix(tp=3, fp=0, fn=0, tn=5)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


# --- percentile ---------------------------------------------------------------------


def test_percentile_nearest_rank():
    values = [50.0, 10.0, 30.0, 40.0, 20.0]
    assert percentile(values, 50) == 30.0
    assert percentile(values, 95) == 50.0
    assert percentile(values, 0) == 10.0


def test_percentile_empty():
    assert percentile([], 95) is None


# --- faithfulness aggregation -------------------------------------------------------


def test_aggregate_faithfulness_counts():
    agg = aggregate_faithfulness([(2, 2), (3, 1)])
    assert agg.fully_grounded == 1          # only the (2,2) answer had zero unsupported
    assert agg.fully_grounded_rate == 0.5
    assert agg.total_claims == 5
    assert agg.supported_claims == 3
    assert agg.claim_support_rate == 0.6


def test_aggregate_faithfulness_zero_claims_is_vacuously_grounded():
    agg = aggregate_faithfulness([(0, 0)])
    assert agg.fully_grounded == 1
    assert agg.claim_support_rate is None   # no claims -> claim-level undefined


# --- judge output parsing -----------------------------------------------------------


def test_parse_verdicts_fenced_json():
    raw = '```json\n[{"claim":"a","verdict":"SUPPORTED"},{"claim":"b","verdict":"NOT_SUPPORTED"}]\n```'
    verdicts = _parse_verdicts(raw)
    assert [v.supported for v in verdicts] == [True, False]


def test_parse_verdicts_bare_array_with_prose():
    raw = 'Here is the result: [{"claim":"x","verdict":"supported"}] done'
    verdicts = _parse_verdicts(raw)
    assert len(verdicts) == 1 and verdicts[0].supported is True


# --- runner ⟷ build_answer consistency ---------------------------------------------
# The whole harness rests on this: evaluate_from_results + abstains_at must reproduce
# build_answer's verdict on the same inputs, or the reported metrics describe a different
# pipeline than the one that ships.


def _verdict_matches(results, generator):
    reason, confidence, _ = evaluate_from_results("q", results, generator)
    runner_abstains = abstains_at(reason, confidence, _SETTINGS.confidence_abstain_threshold)
    built = build_answer("q", results, generator, settings=_SETTINGS)
    return runner_abstains == (built.verdict is AnswerVerdict.ABSTAINED)


def test_consistency_structural_abstain():
    reason, _, _ = evaluate_from_results("q", [], _FakeGenerator(GenResult("x")))
    assert reason is Reason.STRUCTURAL
    assert _verdict_matches([], _FakeGenerator(GenResult("x")))


def test_consistency_sentinel_abstain():
    results = [_result("a", "c", 0.03, dense=0.8), _result("b", "d", 0.02, dense=0.2)]
    gen = _FakeGenerator(GenResult(text="nope", abstained=True))
    reason, _, _ = evaluate_from_results("q", results, gen)
    assert reason is Reason.SENTINEL
    assert _verdict_matches(results, gen)


def test_consistency_scored_answer_clear_winner():
    results = [
        _result("a", "alpha", 0.031, dense=0.80),
        _result("b", "bravo", 0.030, dense=0.20),
        _result("c", "charlie", 0.029, dense=0.18),
    ]
    reason, conf, _ = evaluate_from_results("q", results, ExtractiveGenerator())
    assert reason is Reason.SCORED and conf >= 0.12
    assert _verdict_matches(results, ExtractiveGenerator())


def test_consistency_scored_low_confidence_abstain():
    results = [
        _result("a", "x", 0.031, dense=0.44),
        _result("b", "y", 0.030, dense=0.43),
        _result("c", "z", 0.029, dense=0.43),
    ]
    reason, conf, _ = evaluate_from_results("q", results, ExtractiveGenerator())
    assert reason is Reason.SCORED and conf < 0.12
    assert _verdict_matches(results, ExtractiveGenerator())


# --- dataset loads and is internally consistent -------------------------------------


def test_bundled_dataset_parses_and_is_labeled_consistently():
    cases = load_dataset("evaluation/datasets/support_qa.jsonl")
    assert len(cases) >= 20
    for c in cases:
        if c.category == "answerable":
            assert c.gold_source and not c.expected_abstain
        if c.category in ("ambiguous", "out_of_scope"):
            assert c.gold_source is None and c.expected_abstain
        if c.category == "unanswerable_on_topic":
            assert c.gold_source and c.expected_abstain
