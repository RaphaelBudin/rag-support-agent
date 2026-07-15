"""Blind-spot detection + observability (M7) — closing the loop.

Every served query is written to an append-only ``query_events`` log. Two views fall out
of that one table:

  * **Observability** — cost, latency, and token counts over *all* traffic
    (:func:`observability_summary`).
  * **Blind-spot detection** — the subset we could *not* confidently answer (``abstained``,
    i.e. M3's no-source refusal ∪ the sentinel refusal ∪ M4's low-confidence abstention),
    clustered into a **knowledge-gap report**: "the top things users ask that we can't
    answer well" — which is exactly the list of docs to write next.

Contrast with ``knowledge_units`` (idempotent upsert): this log is *append-only*. A question
asked ten times is not a duplicate to collapse — that repetition is the whole signal (ten
people hit the same gap), so we keep every row and let ``count`` carry the volume.

The clean-room hard part is **clustering "what we can't answer" without an LLM**. Keyless we
can only group by *shared vocabulary* (:func:`cluster_lexical`, over the very same
error-code-preserving tokenizer BM25 ranks on); grouping by *meaning* needs an embedding
(:func:`cluster_semantic`), the optional path gated on an embedding provider — the same
keyless-coarse-vs-gated pattern as M2/M5. The honest limit is stated on the tin: lexical
cannot merge paraphrases with disjoint words ("export my data" vs "get a copy of everything
I've stored"); only the semantic path joins those.

Privacy note: a real deployment logs *user* queries, which is PII-sensitive. Here every query
is synthetic (the 23-case eval set), so the log carries nothing personal — but the schema is
where a retention/redaction policy would live in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import psycopg

from rag_support_agent.config import Settings, get_settings
from rag_support_agent.eval.metrics import percentile
from rag_support_agent.knowledge.db import get_conn
from rag_support_agent.retrieval.embeddings import Embedder, get_embedder
from rag_support_agent.retrieval.hybrid import tokenize


@dataclass
class QueryEvent:
    """One served query's telemetry, appended to the log.

    ``cost_usd`` is the per-request serving cost (tokens × the running provider's list
    price); ``input_tokens``/``output_tokens`` are kept alongside it so the log carries the
    raw counts the cost is derived from, not just the money. ``at`` defaults to the DB clock
    on insert when left ``None``.
    """

    query: str
    confidence: float
    abstained: bool
    top_source: str | None
    latency_ms: float
    cost_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    at: datetime | None = None


# --------------------------------------------------------------------------- persistence


def init_events_schema(conn: psycopg.Connection) -> None:
    """Create the append-only ``query_events`` table if it doesn't exist (idempotent)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS query_events (
                id            BIGSERIAL PRIMARY KEY,
                query         TEXT NOT NULL,
                confidence    DOUBLE PRECISION NOT NULL,
                abstained     BOOLEAN NOT NULL,
                top_source    TEXT,
                latency_ms    DOUBLE PRECISION,
                cost_usd      DOUBLE PRECISION,
                input_tokens  INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                at            TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
    conn.commit()


def _insert_event(conn: psycopg.Connection, event: QueryEvent) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO query_events
                (query, confidence, abstained, top_source, latency_ms, cost_usd,
                 input_tokens, output_tokens, at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()));
            """,
            (
                event.query,
                event.confidence,
                event.abstained,
                event.top_source,
                event.latency_ms,
                event.cost_usd,
                event.input_tokens,
                event.output_tokens,
                event.at,
            ),
        )


def record(
    event: QueryEvent,
    settings: Settings | None = None,
    conn: psycopg.Connection | None = None,
) -> None:
    """Persist a query event.

    Pass ``conn`` (already schema-initialized) to reuse a connection across a batch — the
    replay seeder does this. With no ``conn`` the call self-manages: open, ensure schema,
    insert, commit, close. Either way this is telemetry, so callers keep it best-effort —
    a logging failure must never fail the user's answer (see ``answer.answer_question``).
    """
    if conn is not None:
        _insert_event(conn, event)
        conn.commit()
        return
    c = get_conn(settings)
    try:
        init_events_schema(c)
        _insert_event(c, event)
        c.commit()
    finally:
        c.close()


def _fetch_unanswered(conn: psycopg.Connection) -> list[tuple[str, str | None]]:
    """Every abstained query with its closest source, oldest first (duplicates kept)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT query, top_source FROM query_events WHERE abstained ORDER BY at, id;"
        )
        return [(q, src) for q, src in cur.fetchall()]


# --------------------------------------------------------------------------- clustering (pure)


def salient_terms(query: str) -> list[str]:
    """Unique salient terms of a query, first-appearance order (the retrieval tokenizer)."""
    return list(dict.fromkeys(tokenize(query)))


@dataclass
class GapCluster:
    """One theme of unanswered questions: a label, how often it was hit, and where to look."""

    theme: str
    count: int
    example: str
    terms: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Attach the higher root under the lower so component ids stay deterministic.
            self.parent[max(ra, rb)] = min(ra, rb)

    def components(self, n: int) -> list[list[int]]:
        comps: dict[int, list[int]] = {}
        for i in range(n):
            comps.setdefault(self.find(i), []).append(i)
        out = [sorted(members) for members in comps.values()]
        out.sort(key=lambda m: (-len(m), m[0]))  # biggest first, then smallest index
        return out


def cluster_lexical(queries: list[str]) -> list[list[int]]:
    """Group query indices into connected components that share ≥1 salient term.

    Single-linkage over the shared-vocabulary graph: two queries are linked if their salient
    terms intersect, and transitively merged. Order-independent and deterministic. This is
    the keyless-coarse clusterer — it groups "how do I export data" with "exporting fails"
    (shared ``export``) but *cannot* reach the paraphrase with no shared word; and, being
    single-linkage, it can chain (A–B on one term, B–C on another) — both limits are the
    price of having no embedding, and are stated in the write-up.
    """
    term_sets = [set(salient_terms(q)) for q in queries]
    uf = _UnionFind(len(queries))
    first_seen: dict[str, int] = {}
    for i, terms in enumerate(term_sets):
        for t in terms:
            if t in first_seen:
                uf.union(first_seen[t], i)
            else:
                first_seen[t] = i
    return uf.components(len(queries))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def cluster_semantic(
    queries: list[str], embedder: Embedder, threshold: float = 0.62
) -> list[list[int]]:
    """Group query indices by embedding cosine ≥ ``threshold`` (single-linkage).

    The gated path (needs a real embedder — hash vectors carry no meaning, so they cannot
    join paraphrases). Same connected-component shape as :func:`cluster_lexical` so the
    caller is agnostic to which ran. This is what catches the disjoint-vocabulary paraphrase
    the lexical clusterer structurally cannot. The default ``threshold`` is the measured
    Gemini-query band (``config.gap_semantic_threshold``); it is embedder-specific.
    """
    if not queries:
        return []
    vecs = embedder.embed(queries)
    uf = _UnionFind(len(queries))
    for i in range(len(queries)):
        for j in range(i + 1, len(queries)):
            if _cosine(vecs[i], vecs[j]) >= threshold:
                uf.union(i, j)
    return uf.components(len(queries))


def _build_clusters(
    queries: list[str],
    sources: list[str | None],
    components: list[list[int]],
) -> list[GapCluster]:
    """Turn index components into labelled, counted, example-carrying gap clusters.

    Theme = the salient term(s) shared by the most member questions (frequency desc, then
    alphabetical), so a cluster is named by its common vocabulary. Example = the shortest
    member question (the most canonical phrasing). Count = member questions, duplicates
    included (volume = how many times the gap was hit).
    """
    clusters: list[GapCluster] = []
    for members in components:
        qs = [queries[i] for i in members]
        term_freq: dict[str, int] = {}
        for q in qs:
            for t in salient_terms(q):
                term_freq[t] = term_freq.get(t, 0) + 1
        terms = sorted(term_freq, key=lambda t: (-term_freq[t], t))
        theme = " ".join(terms[:2]) if terms else min(qs, key=lambda q: (len(q), q))
        example = min(qs, key=lambda q: (len(q), q))
        srcs = list(dict.fromkeys(sources[i] for i in members if sources[i]))
        clusters.append(
            GapCluster(theme=theme, count=len(members), example=example, terms=terms, sources=srcs)
        )
    clusters.sort(key=lambda c: (-c.count, c.theme))
    return clusters


def gap_clusters(
    limit: int = 10,
    settings: Settings | None = None,
    conn: psycopg.Connection | None = None,
    mode: str = "auto",
) -> list[GapCluster]:
    """The knowledge-gap clusters, richest form: fetch abstained queries → cluster → rank.

    ``mode`` selects the clusterer: ``"lexical"`` (keyless), ``"semantic"`` (embedding-gated),
    or ``"auto"`` — semantic when an embedding provider is configured (``embedding_provider``
    ≠ ``hash``), lexical otherwise. Same auto-detect pattern the rest of the repo uses.
    """
    s = settings or get_settings()
    own = conn is None
    c = conn or get_conn(s)
    try:
        init_events_schema(c)
        rows = _fetch_unanswered(c)
    finally:
        if own:
            c.close()
    if not rows:
        return []
    queries = [q for q, _ in rows]
    sources = [src for _, src in rows]

    if mode not in ("auto", "lexical", "semantic"):
        raise ValueError(f"unknown gap-report mode: {mode!r} (use auto|lexical|semantic)")
    use_semantic = mode == "semantic" or (mode == "auto" and s.embedding_provider != "hash")
    components = (
        cluster_semantic(queries, get_embedder(s), s.gap_semantic_threshold)
        if use_semantic
        else cluster_lexical(queries)
    )
    return _build_clusters(queries, sources, components)[:limit]


def knowledge_gap_report(
    limit: int = 10,
    settings: Settings | None = None,
    conn: psycopg.Connection | None = None,
) -> list[tuple[str, int]]:
    """Top unanswered/low-confidence query clusters -> ``(theme, count)``, most-hit first."""
    return [(c.theme, c.count) for c in gap_clusters(limit, settings=settings, conn=conn)]


# --------------------------------------------------------------------------- observability


@dataclass
class ObservabilitySummary:
    """Per-request cost/latency/token rollup over the whole log."""

    n_events: int
    n_abstained: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    total_cost_usd: float
    mean_cost_usd: float | None
    mean_input_tokens: float | None
    mean_output_tokens: float | None

    @property
    def abstain_rate(self) -> float | None:
        return self.n_abstained / self.n_events if self.n_events else None


def observability_summary(
    settings: Settings | None = None, conn: psycopg.Connection | None = None
) -> ObservabilitySummary:
    """Roll up latency percentiles, cost, and token means across every logged request."""
    s = settings or get_settings()
    own = conn is None
    c = conn or get_conn(s)
    try:
        init_events_schema(c)
        with c.cursor() as cur:
            cur.execute(
                "SELECT abstained, latency_ms, cost_usd, input_tokens, output_tokens "
                "FROM query_events;"
            )
            rows = cur.fetchall()
    finally:
        if own:
            c.close()

    n = len(rows)
    if n == 0:
        return ObservabilitySummary(0, 0, None, None, 0.0, None, None, None)
    latencies = [r[1] for r in rows if r[1] is not None]
    costs = [r[2] or 0.0 for r in rows]
    return ObservabilitySummary(
        n_events=n,
        n_abstained=sum(1 for r in rows if r[0]),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        total_cost_usd=sum(costs),
        mean_cost_usd=sum(costs) / n,
        mean_input_tokens=sum(r[3] or 0 for r in rows) / n,
        mean_output_tokens=sum(r[4] or 0 for r in rows) / n,
    )
