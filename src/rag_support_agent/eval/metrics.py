"""Pure metric functions for the evaluation harness — no DB, no network.

Kept separate from ``run`` (which does the DB-backed orchestration) so every number
the harness reports is computed by a function you can unit-test with fabricated inputs
(``tests/test_eval``). That is the whole point of an eval harness: the metrics
themselves have to be trustworthy before the scores they produce mean anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def normalize_source(uri: str) -> str:
    """Reduce a source label to a comparable key: its basename.

    Gold labels are authored as ``api-keys.md`` (the convention the loader actually
    stores, ``source_uri = path.relative_to(source_dir)``), but a stray
    ``sample_docs/api-keys.md`` or an absolute path should still match. Comparing on
    basename makes Recall robust to how the path was written, not what it points at.
    """
    return os.path.basename(uri.strip())


def _gold_set(gold: str | list[str] | None) -> set[str]:
    if gold is None:
        return set()
    items = [gold] if isinstance(gold, str) else gold
    return {normalize_source(g) for g in items}


def recall_at_k(
    gold_source: str | list[str] | None, retrieved_sources: list[str], k: int
) -> bool | None:
    """Is a gold source among the top-``k`` retrieved? ``None`` if there is no gold.

    Returns a tri-state so the caller can *exclude* no-gold questions (out-of-scope /
    ambiguous) from the Recall denominator instead of scoring them 0 — Recall@k is only
    meaningful for questions that have a right source to find.
    """
    gold = _gold_set(gold_source)
    if not gold:
        return None
    top = {normalize_source(s) for s in retrieved_sources[:k]}
    return bool(gold & top)


@dataclass
class ConfusionMatrix:
    """Confusion matrix for a binary decision. Positive class = *abstain*."""

    tp: int = 0  # abstained AND should have (correct refusal)
    fp: int = 0  # abstained BUT should have answered (over-abstention)
    fn: int = 0  # answered   BUT should have abstained (confident wrong answer — the costly one)
    tn: int = 0  # answered   AND should have (correct answer)

    @property
    def precision(self) -> float | None:
        """Of the times it abstained, how many were right to. ``None`` if it never abstained."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else None

    @property
    def recall(self) -> float | None:
        """Of the questions that should abstain, how many did. ``None`` if none should."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return None
        return 2 * p * r / (p + r)


def abstention_confusion(
    outcomes: list[tuple[bool, bool]],
) -> ConfusionMatrix:
    """Build the confusion matrix from ``(predicted_abstain, expected_abstain)`` pairs."""
    m = ConfusionMatrix()
    for predicted, expected in outcomes:
        if predicted and expected:
            m.tp += 1
        elif predicted and not expected:
            m.fp += 1
        elif not predicted and expected:
            m.fn += 1
        else:
            m.tn += 1
    return m


def percentile(values: list[float], p: float) -> float | None:
    """The ``p``-th percentile (0..100) via nearest-rank. ``None`` for an empty list.

    Nearest-rank (not interpolated) is the honest choice at small N: with ~15 latency
    samples an interpolated p95 invents a value between two measurements, while
    nearest-rank returns a value we actually observed.
    """
    if not values:
        return None
    ordered = sorted(values)
    if p <= 0:
        return ordered[0]
    rank = -(-len(ordered) * p // 100)  # ceil(len*p/100), 1-based
    idx = min(int(rank), len(ordered)) - 1
    return ordered[idx]


@dataclass
class FaithfulnessAggregate:
    """Aggregate of per-answer faithfulness judgements."""

    judged_answers: int
    fully_grounded: int          # answers with zero unsupported claims
    total_claims: int
    supported_claims: int

    @property
    def fully_grounded_rate(self) -> float | None:
        """Headline metric: % of answers with no hallucinated (unsupported) claim."""
        return self.fully_grounded / self.judged_answers if self.judged_answers else None

    @property
    def claim_support_rate(self) -> float | None:
        """Finer view: fraction of individual claims that were supported by context."""
        return self.supported_claims / self.total_claims if self.total_claims else None


def aggregate_faithfulness(per_answer: list[tuple[int, int]]) -> FaithfulnessAggregate:
    """Aggregate ``(n_claims, n_supported)`` pairs, one per judged answer.

    An answer with zero extracted claims is vacuously fully grounded (there is nothing
    to hallucinate) and contributes nothing to the claim-level denominator.
    """
    fully = sum(1 for n_claims, n_sup in per_answer if n_sup >= n_claims)
    total_claims = sum(n_claims for n_claims, _ in per_answer)
    supported = sum(n_sup for _, n_sup in per_answer)
    return FaithfulnessAggregate(
        judged_answers=len(per_answer),
        fully_grounded=fully,
        total_claims=total_claims,
        supported_claims=supported,
    )
