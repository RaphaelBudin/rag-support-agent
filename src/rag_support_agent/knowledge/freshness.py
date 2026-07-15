"""Knowledge freshness / decay scoring (M6).

Docs rot: yesterday's correct answer is today's wrong one. This module scores how
*likely* a retrieved source is to have gone stale and flags the risky ones, so an
answer can carry a "possibly stale" warning and a human knows which docs to
re-verify first.

The honest hard part — **there is no ground truth.** Nothing in the corpus says "this
doc is wrong now." All we have is *when the source was last updated*
(``KnowledgeUnit.source_updated_at``, the file mtime captured at ingest). Age is a
*proxy for decay risk*, not a measurement of staleness — a 5-year-old doc for a stable
API is fine; a 2-day-old doc can already be wrong. So this is a triage signal, never a
correctness verdict, and we say so.

Two complementary signals, mirroring M4's lesson (a *relative* signal transfers across
scales; an *absolute* floor doesn't):

  1. **Absolute decay** — an exponential half-life over age: ``0.5 ** (age / half_life)``.
     Catches "the whole corpus is old." The half-life is a policy knob (weeks for a
     pricing page, years for a legal doc), not something we can learn keyless.
  2. **Relative outlier** — a source much older than the *other retrieved sources*.
     Catches "this one lags its peers" even inside a fresh absolute band, and — crucially
     — stays silent when the corpus is uniform. A fresh ``git clone`` stamps every file
     with the clone time, so absolute age collapses to ~0 for everything; the relative
     signal then correctly flags *nothing* instead of firing arbitrarily.

A source is flagged when *either* signal fires. Everything here is pure over timestamps
+ an injectable ``now`` — no DB, no API — so ``build_answer`` stays keyless and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.knowledge.models import RetrievalResult

_DAY_SECONDS = 86_400.0


def freshness_score(age_days: float, half_life_days: float) -> float:
    """Absolute freshness in ``(0, 1]`` from age via exponential half-life decay.

    ``score = 0.5 ** (age_days / half_life_days)``: 1.0 at age 0, 0.5 at one half-life,
    0.25 at two, asymptoting toward 0. Negative ages (clock skew) clamp to 0 → 1.0, and a
    non-positive half-life disables decay (everything reads perfectly fresh) rather than
    dividing by zero.
    """
    if half_life_days <= 0:
        return 1.0
    return 0.5 ** (max(0.0, age_days) / half_life_days)


@dataclass
class FreshnessReport:
    """Per-source freshness for one retrieved set.

    ``scores``/``ages_days`` cover every distinct source that carried a timestamp;
    ``stale_sources`` is the subset flagged possibly-stale (sorted), and ``reasons`` maps
    each of those to which signal fired (``"absolute"`` / ``"relative"`` / both).
    """

    stale_sources: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    ages_days: dict[str, float] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)


def _age_days(ts: datetime, now: datetime) -> float:
    return (now - ts).total_seconds() / _DAY_SECONDS


def _relative_outliers(ages_days: dict[str, float], settings: Settings) -> set[str]:
    """Sources whose age is a strong outlier above the median of their retrieved peers.

    Needs >= 3 distinct sources for the median to mean anything (with one or two, "older
    than the pack" is just "older", which the absolute signal already handles). A source is
    an outlier if it is both a *multiple* of the median age (``>= factor x median``) and an
    absolute *gap* older (``>= min_gap_days``) — the gap floor stops a corpus of near-equal
    ages from flagging a source that is trivially, harmlessly older than its siblings.
    """
    if len(ages_days) < 3:
        return set()
    med = median(ages_days.values())
    if med <= 0:
        return set()
    factor = settings.freshness_relative_factor
    min_gap = settings.freshness_relative_min_gap_days
    return {
        src
        for src, age in ages_days.items()
        if age >= factor * med and (age - med) >= min_gap
    }


def assess_freshness(
    results: list[RetrievalResult],
    settings: Settings | None = None,
    now: datetime | None = None,
) -> FreshnessReport:
    """Score the retrieved sources' freshness and flag the possibly-stale ones.

    Pure: reads each unit's ``source_updated_at`` against ``now`` (defaults to the current
    UTC time). Units within one source share a single mtime, so we assess per distinct
    ``source_uri``. A source with no timestamp is skipped entirely — *unknown* freshness is
    not the same as stale, so we never flag on missing data.
    """
    s = settings or get_settings()
    now = now or datetime.now(tz=timezone.utc)

    # One age per distinct source (first timestamp seen; a source's units share their mtime).
    per_source: dict[str, datetime] = {}
    for r in results:
        ts = r.unit.source_updated_at
        if ts is not None:
            per_source.setdefault(r.unit.source_uri, ts)

    ages = {src: _age_days(ts, now) for src, ts in per_source.items()}
    scores = {src: freshness_score(age, s.freshness_half_life_days) for src, age in ages.items()}

    absolute = {src for src, sc in scores.items() if sc < s.freshness_stale_score}
    relative = _relative_outliers(ages, s)

    reasons: dict[str, str] = {}
    for src in absolute | relative:
        tags = []
        if src in absolute:
            tags.append("absolute")
        if src in relative:
            tags.append("relative")
        reasons[src] = "+".join(tags)

    return FreshnessReport(
        stale_sources=sorted(reasons),
        scores=scores,
        ages_days=ages,
        reasons=reasons,
    )
