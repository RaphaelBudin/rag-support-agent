# RAG Support Agent

**A production-grade knowledge assistant that answers support questions from your docs — and knows when *not* to answer.**

Most RAG demos retrieve a few chunks, stuff them into a prompt, and hope. This one is built the way a real support copilot has to be built to survive contact with users: it is **evaluated**, it **abstains when it isn't sure**, it **tracks whether its knowledge is going stale**, and it **reports the questions it couldn't answer** so the knowledge base can be improved.

> **Origin.** This is a clean-room reimplementation of an architecture I designed, shipped, and operated in production (a customer-support knowledge assistant that cut manual support tickets by ~76%). This repository runs the same techniques on **public / synthetic data** — it contains no proprietary code or data from any employer.

---

## Why this is not a toy

The hard part of RAG in production isn't retrieval — it's everything around it. This repo implements the parts that usually get skipped:

| Capability | What it does | Why it matters |
|---|---|---|
| **Evaluation harness** | Precision/recall on retrieval + answer faithfulness against a labeled Q/A set | You can't improve what you don't measure. Turns "seems good" into a number. |
| **Confidence + abstention** | Scores answer confidence; says *"I don't know / here's who to ask"* below a threshold | A support bot that hallucinates once loses user trust permanently. |
| **Knowledge freshness / decay** | Tracks source age; flags knowledge units likely to be stale | Docs rot. Yesterday's correct answer is today's wrong answer. |
| **Blind-spot detection** | Logs low-confidence / unanswered queries into a knowledge-gap report | Tells you exactly what to write next. Closes the loop. |
| **Hybrid retrieval** | Vector (pgvector) + keyword (BM25) fusion | Pure vector search misses exact-match terms (error codes, API names). |
| **Cost & latency observability** | Per-request token/cost/latency logging | AI features die in production from silent cost creep. |

---

## Architecture

```
             ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
  docs  ───▶ │  Ingestion  │ ──▶ │  Knowledge   │ ──▶ │    Retrieval    │
 tickets     │ load/chunk  │     │    units     │     │ pgvector + BM25 │
             └─────────────┘     │ + freshness  │     │  hybrid fusion  │
                                 └──────────────┘     └────────┬────────┘
                                                               │
   ┌──────────────┐   ┌────────────────┐   ┌──────────────────▼────────┐
   │  Chat UI     │◀─ │   Generation   │◀─ │   Context assembly        │
   │  (thin)      │   │ answer + cite  │   │   + relevance gate        │
   └──────────────┘   │ + confidence   │   └───────────────────────────┘
                      │ + abstention   │
                      └───────┬────────┘
                              │  low-confidence / no-source
                              ▼
                   ┌──────────────────────┐
                   │  Blind-spot log  ──▶  │  knowledge-gap report
                   │  Observability   ──▶  │  cost / latency / eval
                   └──────────────────────┘
```

## Stack

Python · FastAPI · PostgreSQL + **pgvector** · OpenAI / Claude (pluggable LLM + embedding providers) · BM25 (hybrid) · Docker Compose · a thin chat UI.

---

## Quickstart

```bash
# 1. Bring up Postgres + pgvector
docker compose up -d

# 2. Install
uv sync            # or: pip install -e .

# 3. Ingest the sample docs
python -m rag_support_agent.ingestion.run --source data/sample_docs

# 4. Query it — hybrid retrieval with per-arm scores (M2)
python -m rag_support_agent.retrieval.search --query "401 Unauthorized error" --show-arms

# 6. Run the API + UI  (M3/M8 — not built yet)
python -m rag_support_agent.api.server
# open http://localhost:8000

# 7. Run the eval harness  (M5 — not built yet)
python -m rag_support_agent.eval.run --dataset evaluation/datasets/support_qa.jsonl
```

> **Runs keyless out of the box.** The default embedder (`EMBEDDING_PROVIDER=hash`) is a
> deterministic, no-API-key stand-in so you can try the whole flow in minutes; set it to
> `openai` for real embeddings. Ingestion alone needs no database — try
> `python -m rag_support_agent.ingestion.run --source data/sample_docs --dry-run`.

## Evaluation

The point of the eval harness is that these numbers are **reproducible** — run `eval.run` and you get them yourself.

| Metric | Score | Notes |
|---|---|---|
| Retrieval Recall@5 | _tbd_ | fraction of questions whose gold source is in top-5 |
| Answer faithfulness | _tbd_ | % of answers grounded in retrieved context (no hallucination) |
| Abstention precision | _tbd_ | when it says "I don't know", it should be right to |
| p95 latency | _tbd_ | end-to-end |
| Cost / 1k queries | _tbd_ | |

_(Filled in as the build progresses — see [BUILD-PLAN.md](BUILD-PLAN.md).)_

---

## Design decisions (the interesting part)

Short write-ups of the non-obvious calls — this is where the engineering lives.

### Chunking: heading-aware, section-scoped

**Decision.** Split each document on Markdown headings first — one chunk per section,
tagged with its full heading path (`API keys > Rotating a key`) — and only *window* a
section into overlapping pieces if it exceeds a size target (1200 chars, 150 overlap).

**Why not fixed-size windows** (the default everyone reaches for). Support docs are
already structured by humans into topic-sized sections, and a heading is the strongest
available signal of what a passage is *about*. Fixed-size chunking cuts across that
structure — it welds the tail of "Rotating a key" onto the head of "Revoking a key," so
one chunk half-answers two questions and cites neither cleanly. Heading-first keeps every
chunk inside a single topic and hands it a precise citation (the heading path), which the
generation step later reuses verbatim.

**Challenge.** Some sections are longer than you want in one chunk, and naive splitting
there drops the sentence that straddles the cut. So the windowing fallback packs whole
*paragraphs* (never mid-paragraph) up to the target and carries a word-aligned overlap
tail into the next window — a fact split across the boundary survives in both pieces.

**What I measured.** On the sample corpus, heading-aware chunking produced a clean **1
chunk per section: 28 chunks across 28 distinct (doc, section) pairs**, max 318 chars,
mean 175. Every section fit under the target, so the windowing path never fired on this
corpus (it's exercised by a unit test with a synthetic long section). That 1:1 mapping is
the payoff — every retrieved unit maps to exactly one documentation heading, so citations
are unambiguous and no vector is diluted by two topics.

**Trade-off / what breaks.** Very short sections (the 62-char minimum here) embed into
thin vectors — fine for keyword/BM25, weaker for dense semantic match. If sections were
routinely tiny I'd merge adjacent small siblings under the same parent heading up to a
floor size. This corpus doesn't need it; a larger one might.

### Why hybrid retrieval over pure vector

**Decision.** Run two retrievers — dense (pgvector cosine over embeddings) and sparse
(BM25 keyword) — fuse them with Reciprocal Rank Fusion, then apply a relevance gate.
Not vector-only.

**Why.** Support questions mix two things that reward *opposite* retrievers:
- **Exact tokens** — error codes (`E_RATE_LIMIT`), status codes (`401`), key prefixes
  (`ak_test_`), header names (`Retry-After`). Embeddings blur these into a neighborhood;
  BM25, with IDF weighting, treats a rare exact token as a strong, precise signal.
- **Paraphrase** — "how do I stop a leaked key from working" for a section titled
  *Revoking a key*, sharing no keywords. BM25 scores ~0 here; a semantic embedding scores
  high.

No single retriever is good at both, and which one a query needs isn't known in advance.
So run both and fuse.

**Why RRF and not a weighted score blend.** Cosine similarity sits in ~[0,1]; BM25 scores
are unbounded and corpus-dependent. Blending them means normalizing incomparable scales
and picking a weight that's really query-dependent. RRF sidesteps that: it fuses on
**rank** (`score = Σ 1/(k + rank)`, k=60), so there's no normalization and no weight to
tune. It's the boring, robust default — and it's unit-tested (`tests/test_fusion.py`).

**The relevance gate.** After fusion, a candidate survives only if the dense arm clears an
absolute cosine floor **or** the sparse arm has any keyword hit. Failing both means
out-of-scope, so the query surfaces *nothing* — which is what lets the agent abstain (M4)
instead of returning its least-bad guess. The gate is a coarse pre-filter; calibrated
abstention (score spread + grounding) is a separate, later threshold.

**What I measured** — reproducible, keyless (`EMBEDDING_PROVIDER=hash`, zero setup):

1. **BM25 recovers a relevant source that pure vector discards.** For `"401 Unauthorized
   error"`, the chunk `api-keys.md > Revoking a key` (a revoked key fails with `401
   Unauthorized` / `E_AUTH_INVALID`) scores dense cosine **0.140 — below the 0.15 gate**,
   so vector-only-plus-gate drops it. BM25 ranks it **#2** on the exact `401` /
   `E_AUTH_INVALID` match, and fusion returns it at **#4**. Hybrid keeps a correct source
   that vector alone threw away.
2. **The gate abstains on out-of-scope.** For `"how do I bake sourdough bread"`, BM25
   returns nothing (no keyword overlap) and the best dense cosine is **0.064**, far under
   the floor — the gate returns **empty**. No hallucinated "closest" source.

```
python -m rag_support_agent.retrieval.search --query "401 Unauthorized error" --show-arms
```

**Honest limitation of the keyless demo.** The default `hash` embedder is *lexical*
(feature-hashing of tokens), not semantic — so keyless, the dense arm and BM25 largely
**agree**, and fusion barely reorders them. Hybrid's bigger payoff, recovering pure
*paraphrases* that share no keywords with the query, needs a real embedder. The harness
supports it; reproduce with your own key:
```
EMBEDDING_PROVIDER=openai python -m rag_support_agent.ingestion.run --source data/sample_docs
EMBEDDING_PROVIDER=openai python -m rag_support_agent.retrieval.search \
  --query "how do I make a leaked credential stop working immediately" --show-arms
```

**Also fixed here, found via this measurement.** The dense arm first embedded only the
chunk *body*, so a chunk whose error code lives in its *heading* (`... > E_RATE_LIMIT
(429)`) was near-invisible to vector search — it sat at dense rank **14**. Ingestion now
embeds the heading path together with the body (the stored/cited text stays the pure
body); the same chunk moves to rank **3**. Lesson: embed your headings — they're the
strongest topic label a chunk has.
### Confidence signal: how it's computed and calibrated — _tbd (M4)_
### Knowledge decay: modeling staleness without ground truth — _tbd (M6)_
### Self-hosted pgvector vs a managed vector DB (cost/control trade-off) — _tbd_

## License

GPLv3 — see [LICENSE](LICENSE). Copyleft on purpose: this is a public reference
implementation, and I want derivatives to stay open.
