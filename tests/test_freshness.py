"""Unit tests for knowledge freshness / decay (M6).

All DB-free and key-free: freshness is pure over timestamps + an injected ``now``, so the
decay math, the two-signal flag, and the ``build_answer`` wiring are pinned down without
Postgres or a clock in the loop.
"""

from datetime import datetime, timedelta, timezone

from rag_support_agent.config import Settings
from rag_support_agent.generation.answer import build_answer
from rag_support_agent.generation.generators import ExtractiveGenerator
from rag_support_agent.knowledge.freshness import (
    FreshnessReport,
    assess_freshness,
    freshness_score,
)
from rag_support_agent.knowledge.models import AnswerVerdict, KnowledgeUnit, RetrievalResult

NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def _result(uid: str, age_days: float, dense: float | None = None) -> RetrievalResult:
    """A retrieved unit whose source was last updated ``age_days`` before ``NOW``."""
    unit = KnowledgeUnit(
        id=uid,
        content=f"content of {uid}",
        source_uri=f"{uid}.md",
        section=uid,
        source_updated_at=NOW - timedelta(days=age_days),
    )
    return RetrievalResult(unit=unit, score=0.03, dense_similarity=dense)


# --- freshness_score (pure decay curve) ---------------------------------------------


def test_score_is_one_at_age_zero():
    assert freshness_score(0.0, half_life_days=180.0) == 1.0


def test_score_halves_every_half_life():
    assert freshness_score(180.0, 180.0) == 0.5
    assert freshness_score(360.0, 180.0) == 0.25


def test_score_is_monotonic_decreasing():
    curve = [freshness_score(d, 180.0) for d in (0, 30, 90, 180, 365, 730)]
    assert curve == sorted(curve, reverse=True)


def test_negative_age_clamps_to_fresh():
    # Clock skew (source_updated_at in the future) must not blow past 1.0.
    assert freshness_score(-10.0, 180.0) == 1.0


def test_nonpositive_half_life_disables_decay():
    assert freshness_score(9999.0, 0.0) == 1.0


# --- assess_freshness: absolute signal ----------------------------------------------


def test_fresh_corpus_flags_nothing():
    # Every source a few days old (the sample corpus's real state) -> no flags.
    results = [_result("a", 6), _result("b", 5), _result("c", 4)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert report.stale_sources == []
    assert all(sc > 0.9 for sc in report.scores.values())


def test_source_past_one_half_life_is_absolute_stale():
    results = [_result("old", 400), _result("b", 5), _result("c", 4)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert "old.md" in report.stale_sources
    assert "absolute" in report.reasons["old.md"]
    assert report.scores["old.md"] < 0.5


def test_uniformly_old_corpus_flags_all_absolute_but_none_relative():
    # Absolute catches "everything is old"; relative stays silent (no outlier vs peers).
    results = [_result("a", 730), _result("b", 730), _result("c", 730)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert set(report.stale_sources) == {"a.md", "b.md", "c.md"}
    assert all(report.reasons[src] == "absolute" for src in report.stale_sources)


# --- assess_freshness: relative signal ----------------------------------------------


def test_lone_old_source_is_relative_outlier():
    # One source 2 years old among a fresh field -> flagged on both signals.
    results = [_result("stale", 730), _result("b", 6), _result("c", 6), _result("d", 6)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert report.stale_sources == ["stale.md"]
    assert report.reasons["stale.md"] == "absolute+relative"


def test_uniform_fresh_clone_flags_nothing():
    # A fresh `git clone` resets every mtime to the same instant. Neither signal fires:
    # absolute (all ~fresh) nor relative (no outlier) — the honest "can't distinguish".
    results = [_result("a", 1), _result("b", 1), _result("c", 1), _result("d", 1)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert report.stale_sources == []


def test_relative_needs_at_least_three_sources():
    # Two sources: "older than the pack" is just "older" — leave it to the absolute signal.
    results = [_result("old", 200), _result("b", 6)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert "relative" not in report.reasons.get("old.md", "")


def test_trivially_older_source_is_not_a_relative_outlier():
    # Below the min-gap floor: a few days older than peers must not trip the relative flag.
    results = [_result("a", 12), _result("b", 6), _result("c", 6)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert report.stale_sources == []


def test_missing_timestamp_is_skipped_not_flagged():
    # Unknown freshness != stale: a unit with no source_updated_at is left out entirely.
    unit = KnowledgeUnit(id="x", content="c", source_uri="x.md", source_updated_at=None)
    results = [RetrievalResult(unit=unit, score=0.03, dense_similarity=0.5)]
    report = assess_freshness(results, settings=Settings(), now=NOW)
    assert report == FreshnessReport()


# --- build_answer wiring ------------------------------------------------------------


def _answerable(uid: str, age_days: float, dense: float) -> RetrievalResult:
    r = _result(uid, age_days, dense=dense)
    r.unit.content = f"To do {uid}, follow these steps."
    return r


def test_build_answer_flags_stale_cited_source():
    # Clear-winner spread so it answers; the winning source is 2 years old -> flagged.
    results = [_answerable("a", 730, 0.80), _answerable("b", 6, 0.20), _answerable("c", 6, 0.18)]
    answer = build_answer("q", results, ExtractiveGenerator(), settings=Settings(), now=NOW)
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert "a.md" in answer.stale_sources


def test_build_answer_no_stale_flag_on_fresh_corpus():
    results = [_answerable("a", 6, 0.80), _answerable("b", 5, 0.20), _answerable("c", 4, 0.18)]
    answer = build_answer("q", results, ExtractiveGenerator(), settings=Settings(), now=NOW)
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert answer.stale_sources == []


def test_build_answer_only_flags_cited_sources():
    # 'old_uncited' is stale but ranks below the extractive echo cap (top 3), so it is not a
    # cited/served source -> the flag stays off it. Only sources the reader sees are warned.
    results = [
        _answerable("a", 6, 0.80),
        _answerable("b", 6, 0.20),
        _answerable("c", 6, 0.18),
        _answerable("old_uncited", 730, 0.10),
    ]
    answer = build_answer("q", results, ExtractiveGenerator(), settings=Settings(), now=NOW)
    assert answer.verdict is AnswerVerdict.ANSWERED
    assert "old_uncited.md" not in answer.stale_sources
