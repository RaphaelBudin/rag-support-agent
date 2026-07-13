"""Evaluation harness — one command, real numbers.

    python -m rag_support_agent.eval.run --dataset evaluation/datasets/support_qa.jsonl
    python -m rag_support_agent.eval.run --dataset ... --calibrate      # threshold sweep

Runs the labeled Q/A set end-to-end through the *real* pipeline (hybrid retrieve → gate →
ground → confidence → abstain) and turns "seems good" into the numbers in the README table:

  - Recall@k        : is a gold source in the top-k retrieved? (over questions with a gold)
  - Abstention P/R  : when it abstains, was it right to (precision); of the ones that should
                      abstain, how many did (recall)
  - p50 / p95 latency
  - Faithfulness    : is each answer's every claim entailed by the retrieved context?
  - Cost / 1k       : approximate serving cost from measured generation tokens × list price

Which metrics are real depends on the configured providers, and the harness says so up front:

  - **Keyless** (``EMBEDDING_PROVIDER=hash`` + ``LLM_PROVIDER=extractive``): Recall, abstention,
    latency all compute and are reproducible with no API key — but the *confidence* signal is
    lexical-muted (see M4), so threshold calibration is not meaningful here. Faithfulness is
    grounded-by-construction (the extractive generator echoes verbatim), so it is reported as
    such, not judged.
  - **Gemini-gated** (``EMBEDDING_PROVIDER=gemini`` + ``LLM_PROVIDER=gemini``): the confidence
    signal separates (so ``--calibrate`` is meaningful) and faithfulness is measured by the
    LLM judge. Needs ``GEMINI_API_KEY`` and a Gemini-embedded DB (re-ingest with that provider).

The instrumented runner (``evaluate_from_results``) mirrors ``generation.answer.build_answer``
but exposes the intermediate signals the metrics need (which abstention layer fired, the raw
confidence, the context the generator saw). A test pins its verdict to ``build_answer``'s so
the two cannot drift.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, TypeVar

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.eval.cost import TokenUsage, estimate_cost_usd
from rag_support_agent.eval.faithfulness import Judge, get_judge
from rag_support_agent.eval.metrics import (
    abstention_confusion,
    aggregate_faithfulness,
    percentile,
    recall_at_k,
)
from rag_support_agent.generation.answer import compute_confidence
from rag_support_agent.generation.generators import (
    Generator,
    GenResult,
    format_context,
    get_generator,
)
from rag_support_agent.knowledge.models import RetrievalResult
from rag_support_agent.retrieval.hybrid import retrieve

# Threshold grid for the calibration sweep — the spread signal lives in ~[0, 0.3].
CALIBRATION_GRID = [round(0.01 * i, 2) for i in range(0, 31)]

# Free-tier Gemini caps generation at a low requests/window rate, and the harness fires a
# call per case in a burst — so it self-throttles: on a 429 it backs off for the delay the
# server names (falling back to a default) and retries, bounded. This is exactly the kind of
# quota-awareness a real eval loop needs; without it a 23-case run dies two-thirds through.
_MAX_RETRIES = 5
_DEFAULT_BACKOFF_S = 20.0
# A retryDelay above this means the window will not clear soon (a spent daily bucket, not a
# per-minute burst) — retrying just burns minutes to fail, so we surface it instead.
_DAILY_BACKOFF_THRESHOLD_S = 90.0
# Seconds to wait before each LLM call (set by --pace) to stay under a requests/minute cap.
_PACE_SECONDS = 0.0

T = TypeVar("T")


class DailyQuotaExhausted(RuntimeError):
    """The Gemini free-tier *daily* request cap is spent — backing off cannot help today."""


_DAILY_QUOTA_HELP = (
    "Gemini free-tier DAILY generation cap reached (generate_content, ~20 requests/day). "
    "A full judged eval run needs more calls than that, so it cannot complete on the free "
    "tier in one day. Options: (1) enable billing on the API key to lift the cap, (2) wait "
    "for the daily reset and judge a smaller sample, or (3) run the keyless pass "
    "(LLM_PROVIDER=extractive) — it needs no generation quota. The gemini *embedder* is a "
    "separate quota, so Recall / calibration / abstention still run under EMBEDDING_PROVIDER=gemini."
)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in text or "429" in text


def _backoff_seconds(exc: Exception) -> float:
    """The server's suggested retry delay (``retryDelay: '23s'`` / 'retry in 23s')."""
    m = re.search(r"retry(?:Delay)?[^0-9]*([\d.]+)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else _DEFAULT_BACKOFF_S


def _with_retry(fn: Callable[[], T]) -> T:
    """Call ``fn`` (after an optional pace), backing off a short rate limit; fail fast on a long one.

    The authoritative signal is the server's ``retryDelay``, not the quota label: a per-minute
    burst clears in seconds (honor it and retry), while a spent daily bucket returns a delay far
    in the future (retrying just burns minutes to fail, so surface it instead). Pacing before the
    call keeps a steady rate under the requests/minute cap so bursts don't trip the limit at all.
    """
    if _PACE_SECONDS:
        time.sleep(_PACE_SECONDS)
    for attempt in range(_MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — narrowed by _is_rate_limit, else re-raised
            if not _is_rate_limit(exc) or attempt == _MAX_RETRIES - 1:
                raise
            delay = _backoff_seconds(exc)
            # A single near-full-minute wait can be a per-minute window rolling over. But if we
            # already waited one full window and the *next* delay is still near-full, the window
            # rolled and calls still fail — that is a spent daily bucket, not a burst. Bail then,
            # instead of burning _MAX_RETRIES × ~60s to fail anyway.
            if delay > _DAILY_BACKOFF_THRESHOLD_S or (attempt >= 1 and delay > 45.0):
                raise DailyQuotaExhausted(_DAILY_QUOTA_HELP) from exc
            print(f"  … rate-limited; backing off {delay + 1:.0f}s "
                  f"(attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(delay + 1.0)
    raise RuntimeError("unreachable")  # pragma: no cover


@dataclass
class EvalCase:
    question: str
    category: str
    gold_source: str | list[str] | None
    gold_answer: str | None
    expected_abstain: bool


class Reason(str, Enum):
    """Which abstention layer decided the verdict — the piece ``build_answer`` collapses.

    STRUCTURAL and SENTINEL abstain at *every* threshold (they are threshold-independent);
    only SCORED depends on ``confidence_abstain_threshold``. That split is what lets the
    calibration sweep recompute verdicts from stored records without re-running retrieval.
    """

    STRUCTURAL = "structural"  # Layer 1: gate returned nothing
    SENTINEL = "sentinel"      # Layer 2: generator refused (insufficient context)
    SCORED = "scored"          # generator answered; Layer 3 decides by confidence vs threshold


@dataclass
class EvalRecord:
    case: EvalCase
    reason: Reason
    confidence: float
    retrieved_sources: list[str]
    answer_text: str
    context: str                       # numbered passages the generator saw (for the judge)
    latency_ms: float
    gen_usage: TokenUsage = field(default_factory=TokenUsage)


def load_dataset(path: str | Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cases.append(
            EvalCase(
                question=r["question"],
                category=r.get("category", ""),
                gold_source=r.get("gold_source"),
                gold_answer=r.get("gold_answer"),
                expected_abstain=bool(r["expected_abstain"]),
            )
        )
    return cases


def evaluate_from_results(
    query: str, results: list[RetrievalResult], generator: Generator
) -> tuple[Reason, float, GenResult | None]:
    """Pure decision core (no DB). Mirrors ``build_answer`` but returns the *reason*.

    Empty results → Layer 1 (structural). A generator refusal → Layer 2 (sentinel). Otherwise
    the generator answered and Layer 3 will decide by confidence vs the threshold — so we
    return SCORED with the computed spread and defer the threshold comparison to
    ``abstains_at`` (that is what makes the calibration sweep free).
    """
    if not results:
        return Reason.STRUCTURAL, 0.0, None
    gen = _with_retry(lambda: generator.generate(query, results))
    if gen.abstained:
        return Reason.SENTINEL, 0.0, gen
    return Reason.SCORED, compute_confidence(query, results), gen


def abstains_at(reason: Reason, confidence: float, threshold: float) -> bool:
    """Would the pipeline abstain at this threshold? Matches ``build_answer`` exactly."""
    if reason in (Reason.STRUCTURAL, Reason.SENTINEL):
        return True
    return confidence < threshold


def evaluate_one(case: EvalCase, generator: Generator, settings: Settings) -> EvalRecord:
    """End-to-end one case: retrieve (DB) → decide, timing the whole thing."""
    started = time.perf_counter()
    results = retrieve(case.question, settings=settings)
    reason, confidence, gen = evaluate_from_results(case.question, results, generator)
    latency_ms = (time.perf_counter() - started) * 1000.0
    usage = TokenUsage(
        input_tokens=getattr(generator, "last_input_tokens", 0),
        output_tokens=getattr(generator, "last_output_tokens", 0),
    )
    return EvalRecord(
        case=case,
        reason=reason,
        confidence=confidence,
        retrieved_sources=[r.unit.source_uri for r in results],
        answer_text=gen.text if gen else "",
        context=format_context(results) if results else "",
        latency_ms=latency_ms,
        gen_usage=usage,
    )


# --------------------------------------------------------------------------- metrics assembly


def _recall(records: list[EvalRecord], k: int) -> tuple[int, int]:
    """(#gold-source-in-top-k, #questions-with-a-gold-source)."""
    hits = total = 0
    for rec in records:
        got = recall_at_k(rec.case.gold_source, rec.retrieved_sources, k)
        if got is None:
            continue
        total += 1
        hits += int(got)
    return hits, total


def _abstention(records: list[EvalRecord], threshold: float):
    outcomes = [
        (abstains_at(r.reason, r.confidence, threshold), r.case.expected_abstain)
        for r in records
    ]
    return abstention_confusion(outcomes)


def judge_faithfulness(
    records: list[EvalRecord], judge: Judge, threshold: float
) -> tuple[object, TokenUsage]:
    """Judge every *answered* record (verdict at ``threshold`` is ANSWERED) for grounding.

    Only answered records carry a claim to check; abstentions have no answer to be faithful
    to. Returns the aggregate plus the judge's own token usage (an eval expense, kept separate
    from the serving-cost row).
    """
    per_answer: list[tuple[int, int]] = []
    judge_usage = TokenUsage()
    for rec in records:
        if abstains_at(rec.reason, rec.confidence, threshold):
            continue
        result = _with_retry(lambda rec=rec: judge.judge(rec.answer_text, rec.context))
        per_answer.append((result.n_claims, result.n_supported))
        judge_usage = judge_usage + result.usage
    return aggregate_faithfulness(per_answer), judge_usage


def _serving_cost_per_1k(records: list[EvalRecord]) -> float | None:
    """Approx. serving cost per 1000 queries: mean generation tokens × list price × 1000.

    The judge is an offline eval expense, so it is *excluded* here — this row is what it costs
    to answer a user, not to grade the run. ``None`` when no generation tokens were spent
    (keyless extractive path).
    """
    total = TokenUsage()
    for rec in records:
        total = total + rec.gen_usage
    if total.input_tokens == 0 and total.output_tokens == 0:
        return None
    mean = TokenUsage(
        input_tokens=total.input_tokens // max(len(records), 1),
        output_tokens=total.output_tokens // max(len(records), 1),
    )
    return estimate_cost_usd(mean) * 1000


# --------------------------------------------------------------------------- rendering


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def _print_config_banner(settings: Settings, judge: Judge | None) -> None:
    keyless = settings.llm_provider == "extractive"
    print("=" * 70)
    print("Evaluation run")
    print(
        f"  embedder={settings.embedding_provider}  generator={settings.llm_provider}  "
        f"top_k={settings.retrieval_top_k}  threshold={settings.confidence_abstain_threshold}"
    )
    mode = "KEYLESS (reproducible, confidence muted)" if keyless else "GEMINI-GATED (full suite)"
    print(f"  mode: {mode}")
    print(f"  faithfulness judge: {'on' if judge else 'off (keyless / no key)'}")
    print("=" * 70)


def _print_table(
    records: list[EvalRecord],
    settings: Settings,
    faith,
    faith_note: str,
    cost_per_1k: float | None,
) -> None:
    k = settings.retrieval_top_k
    hits, total = _recall(records, k)
    conf = _abstention(records, settings.confidence_abstain_threshold)
    latencies = [r.latency_ms for r in records]

    recall = hits / total if total else None
    cost_str = f"~${cost_per_1k:.2f} (approx, list price)" if cost_per_1k is not None else "$0 (keyless)"

    print("\nMetric                     Score        Notes")
    print("-" * 70)
    print(f"Retrieval Recall@{k:<9} {_fmt_pct(recall):<12} gold source in top-{k} ({hits}/{total})")
    print(f"Answer faithfulness        {faith:<12} {faith_note}")
    print(
        f"Abstention precision       {_fmt_pct(conf.precision):<12} "
        f"right to abstain ({conf.tp}/{conf.tp + conf.fp})"
    )
    print(
        f"Abstention recall          {_fmt_pct(conf.recall):<12} "
        f"caught the should-abstains ({conf.tp}/{conf.tp + conf.fn})"
    )
    print(f"p50 latency                {percentile(latencies, 50) or 0:.0f} ms")
    print(
        f"p95 latency                {percentile(latencies, 95) or 0:.0f} ms       "
        f"end-to-end, N={len(latencies)}"
    )
    print(f"Cost / 1k queries          {cost_str}")
    print("-" * 70)
    # The costly error class, called out explicitly.
    if conf.fn:
        print(f"  ⚠ {conf.fn} answered-but-should-have-abstained (confident-wrong risk)")
    if conf.fp:
        print(f"  · {conf.fp} abstained-but-answerable (over-abstention)")


def _print_calibration(records: list[EvalRecord], settings: Settings) -> None:
    """Sweep the threshold and print the abstention precision/recall curve.

    Free to compute: every record already carries its reason + confidence, so each threshold
    is just a re-tally (no re-retrieval, no LLM). Only meaningful under a semantic embedder,
    where the confidence signal separates — the banner says which embedder produced these.

    Recommendation = the F1-optimal band (balanced precision/recall). We deliberately do *not*
    chase full abstention recall: one stubborn should-abstain case can sit at a high confidence,
    and the threshold needed to catch it forces abstaining on many good answers — so we report
    that full-recall point and its cost, but recommend the balanced band.
    """
    print("\nThreshold calibration (abstention precision/recall vs confidence_abstain_threshold)")
    print("  τ      precision   recall     F1        answered  abstained")
    print("  " + "-" * 62)
    rows = []
    for tau in CALIBRATION_GRID:
        conf = _abstention(records, tau)
        rows.append((tau, conf))
        print(
            f"  {tau:<6.2f} {_fmt_pct(conf.precision):<11} {_fmt_pct(conf.recall):<10} "
            f"{_fmt_pct(conf.f1):<9} {conf.tn + conf.fn:<9} {conf.tp + conf.fp}"
        )
    print("  " + "-" * 62)

    scored = [(tau, c) for tau, c in rows if c.f1 is not None]
    if not scored:
        print("  → no threshold produces a usable precision/recall trade-off on this run.")
        return
    best_f1 = max(c.f1 for _, c in scored)
    band = [tau for tau, c in scored if abs(c.f1 - best_f1) < 1e-9]
    lo, hi = min(band), max(band)
    current = settings.confidence_abstain_threshold
    in_band = lo <= current <= hi
    print(
        f"  → F1 peaks at {_fmt_pct(best_f1)} across τ∈[{lo:.2f}, {hi:.2f}]. "
        f"Current threshold {current} {'sits in' if in_band else 'is OUTSIDE'} that band."
    )
    full = [(tau, c) for tau, c in rows if c.recall == 1.0]
    if full:
        tau_fr, c_fr = full[0]  # lowest threshold reaching full recall
        print(
            f"  → full abstention recall needs τ≥{tau_fr:.2f}, but that drops precision to "
            f"{_fmt_pct(c_fr.precision)} and answers only {c_fr.tn + c_fr.fn}/{len(records)} "
            f"— over-abstention to catch one outlier; not worth it."
        )


# --------------------------------------------------------------------------- entrypoint


def run(
    dataset: str,
    calibrate: bool,
    no_faithfulness: bool,
    settings: Settings,
    pace: float = 0.0,
) -> None:
    global _PACE_SECONDS
    _PACE_SECONDS = pace
    cases = load_dataset(dataset)
    generator = get_generator(settings)
    judge = None if no_faithfulness else get_judge(settings)
    _print_config_banner(settings, judge)
    print(f"\nRunning {len(cases)} cases...")

    records = [evaluate_one(c, generator, settings) for c in cases]

    if judge is not None:
        faith_agg, judge_usage = judge_faithfulness(
            records, judge, settings.confidence_abstain_threshold
        )
        faith = _fmt_pct(faith_agg.fully_grounded_rate)
        faith_note = (
            f"answers with 0 unsupported claims ({faith_agg.fully_grounded}/"
            f"{faith_agg.judged_answers}); claim-level {_fmt_pct(faith_agg.claim_support_rate)}"
        )
        judge_cost = estimate_cost_usd(judge_usage)
    else:
        faith = "100%*"
        faith_note = "*grounded by construction (extractive echo) — not LLM-judged; set LLM_PROVIDER=gemini"
        judge_cost = 0.0

    cost_per_1k = _serving_cost_per_1k(records)
    _print_table(records, settings, faith, faith_note, cost_per_1k)

    if judge is not None and judge_cost:
        print(f"\n(eval-only) faithfulness judge cost this run: ~${judge_cost:.4f}")

    if calibrate:
        _print_calibration(records, settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent against a labeled set.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL eval set.")
    parser.add_argument(
        "--calibrate", action="store_true", help="Also sweep the abstention threshold."
    )
    parser.add_argument(
        "--no-faithfulness", action="store_true", help="Skip the LLM faithfulness judge."
    )
    parser.add_argument(
        "--pace", type=float, default=0.0,
        help="Seconds to wait before each LLM call (stay under a requests/minute cap).",
    )
    args = parser.parse_args()
    run(args.dataset, args.calibrate, args.no_faithfulness, get_settings(), pace=args.pace)


if __name__ == "__main__":
    main()
