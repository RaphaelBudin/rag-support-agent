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
| **Grounded generation + citations** | Answers only from retrieved context, with inline `[n]` citations; refuses parametric memory | An answer you can't trace to a source is an answer you can't trust — or debug. |
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

Python · FastAPI · PostgreSQL + **pgvector** · Gemini + a keyless *extractive* fallback (pluggable LLM + embedding providers) · BM25 (hybrid) · Docker Compose · a thin chat UI.

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

# 5. Ask it — grounded answer with inline citations (M3)
python -m rag_support_agent.generation.ask --query "How do I rotate an API key?"
# real synthesis instead of verbatim excerpts (needs GEMINI_API_KEY in .env):
LLM_PROVIDER=gemini python -m rag_support_agent.generation.ask --query "How do I rotate an API key?"

# 6. Run the API + UI  (M8 — not built yet)
python -m rag_support_agent.api.server
# open http://localhost:8000

# 7. Run the eval harness  (M5 — not built yet)
python -m rag_support_agent.eval.run --dataset evaluation/datasets/support_qa.jsonl
```

> **Runs keyless out of the box.** Both the default embedder (`EMBEDDING_PROVIDER=hash`) and
> the default generator (`LLM_PROVIDER=extractive`) are no-API-key stand-ins, so you can try
> the whole flow — ingest → retrieve → grounded, cited answer → abstention — in minutes; set
> `EMBEDDING_PROVIDER=gemini` / `LLM_PROVIDER=gemini` for real embeddings and synthesis.
> Ingestion alone needs no database — try
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

**The semantic half — measured with real embeddings (Gemini).** The default `hash`
embedder is *lexical* (feature-hashing of tokens), so keyless the dense arm and BM25
largely **agree** and fusion barely reorders them. Swap in a semantic embedder and
hybrid's bigger payoff appears — recovering pure *paraphrases* that share no keywords
with the query. Query `"how do I make a leaked credential stop working immediately"`
(no words in common with the section *Revoking a key*), with
`EMBEDDING_PROVIDER=gemini` (`gemini-embedding-001`, 1536-d):

- **Dense ranks `api-keys.md > Revoking a key` #1 (cosine 0.601); BM25 doesn't return it
  at all.** Pure keyword search misses the canonical answer, the vector finds it, hybrid
  keeps it (fused #3). This is the case the lexical `hash` embedder *cannot* show.

```
EMBEDDING_PROVIDER=gemini python -m rag_support_agent.ingestion.run --source data/sample_docs
EMBEDDING_PROVIDER=gemini python -m rag_support_agent.retrieval.search \
  --query "how do I make a leaked credential stop working immediately" --show-arms
```

**A gotcha I measured: the cosine floor does not transfer across embedders.** The 0.15
gate floor makes the gate abstain on `"how do I bake sourdough bread"` under `hash` (best
cosine 0.064). Under Gemini the *same* out-of-scope query scores cosine **~0.44 on
unrelated sections** — Gemini packs everything into a high, compressed band — so a 0.15
floor lets it straight through. Absolute cosine thresholds are provider-specific and need
calibration (that's M5). It's also *why* real abstention (M4) keys off score **spread**
(the gap between the top hit and the rest), which is robust across embedders, rather than
an absolute floor: for the out-of-scope query every hit sits at ~0.43 with no clear
winner, whereas a real hit tops out at 0.60–0.70 with separation.

> **Gemini normalization note.** `gemini-embedding-001` only pre-normalizes its full
> 3072-d output; at the 1536-d we request (to match the pgvector column) vectors return
> with L2-norm ~0.7, so `GeminiEmbedder` L2-normalizes them before storage.

**Also fixed here, found via this measurement.** The dense arm first embedded only the
chunk *body*, so a chunk whose error code lives in its *heading* (`... > E_RATE_LIMIT
(429)`) was near-invisible to vector search — it sat at dense rank **14**. Ingestion now
embeds the heading path together with the body (the stored/cited text stays the pure
body); the same chunk moves to rank **3**. Lesson: embed your headings — they're the
strongest topic label a chunk has.
### Grounding & citations: how the agent is stopped from answering off-context

**Decision.** Answer *only* from retrieved context, attach inline `[n]` citations, and
**abstain** rather than guess. Grounding is enforced in **two independent layers**, and the
generator is pluggable — a keyless `extractive` default and a real `gemini` synthesizer —
behind one interface (mirroring the embedder).

**Why two layers, not one prompt.** A support bot that hallucinates *once* loses user trust
permanently, so the design goal isn't "good answers" — it's "never a confident answer from
parametric memory." A single "use only the context" line in a prompt is a request, not a
guarantee. So:

- **Layer 1 — structural (provider-independent, un-foolable).** If the M2 relevance gate
  returns nothing, the query is out of scope, so the generator is *never called* — the agent
  abstains. This is the gate from the previous section closing the loop into a refusal; it
  holds identically under `extractive` and `gemini` because no model runs.
- **Layer 2 — synthesis.** `ExtractiveGenerator` (the keyless default) grounds *by
  construction*: it echoes the top retrieved passages verbatim, each `[n]`-tagged, so it
  physically cannot invent — the strongest grounding guarantee there is, and it's why the
  repo runs end-to-end for a stranger with no key. `GeminiGenerator` grounds *by prompt*:
  temperature 0, "use only the numbered context, cite every claim," and a refusal sentinel
  (`INSUFFICIENT_CONTEXT`) it emits when the context doesn't answer.

**Why the sentinel, when Layer 1 already abstains on empty retrieval.** The gate is coarse —
a single keyword hit passes it. A passage can clear the gate and still not answer the
question. The sentinel is the model *refusing parametric memory while on topic* — that's M3
grounding, and it's deliberately distinct from M4's calibrated, confidence-threshold
abstention (score spread + a real grounding check).

**Challenge — citations have to be stable handles, not a re-count.** Passages are numbered
`[1..n]`; the model cites what it uses. On `"How do I rotate an API key?"` Gemini cited `[1]`
and `[4]`, *skipping* `[2]` and `[3]`. My first cut re-enumerated the citation list to
`[1],[2]` — so the prose said `[4]` while the list said `[2]`, and a reader couldn't reconcile
them. The fix: a `Citation` carries its **true marker index**, so the number in the prose and
the number in the list always agree. Both generators share one numbering (`format_context`)
and one extractor (`parse_citations`), so a `[n]` means the same passage no matter who wrote
it — and a hallucinated out-of-range marker (`[9]` when `n=5`) is dropped, never turned into a
dangling citation.

**A deliberate knob: thinking is off.** `gemini-2.5-flash` ships with a thinking budget;
grounded extraction over supplied passages isn't a reasoning-heavy task, so I set
`thinking_budget=0`. That buys determinism and roughly halves latency/tokens (an on-topic
sentinel refusal costs **6 output tokens**). If the task were multi-hop reasoning *across*
passages, I'd turn it back on.

**What I measured** (real Gemini, `EMBEDDING_PROVIDER=hash` retrieval so it's reproducible):

| Query | Verdict | Citations | Tokens (in/out) | Latency |
|---|---|---|---|---|
| `How do I rotate an API key?` | answered | `[1]` (Rotating a key) + `[4]` (Best practices) | 455 / 101 | ~1.0 s |
| `What does E_RATE_LIMIT mean and how do I fix it?` | answered | `[1] [2]` `errors.md` **+ `[3]` `billing.md`** | 416 / 137 | ~1.1 s |
| `What is the per-call overage rate in USD?` | **abstained** (sentinel) | — | 472 / **6** | ~0.6 s |
| `how do I bake sourdough bread` | **abstained** (structural) | — | *no LLM call* | — |

- **Multi-source grounding.** The `E_RATE_LIMIT` answer fuses `errors.md` (`[1] [2]`) with
  `billing.md`'s spend-cap note (`[3]`) — a genuinely cross-document citation, each claim
  tagged to its source.
- **Refusing to invent a number.** `"per-call overage rate in USD?"` passes the gate on the
  keyword *overage rate* (`billing.md`), but that doc says the number "is shown on the pricing
  page" — it isn't in the context. Gemini emits `INSUFFICIENT_CONTEXT` instead of guessing a
  figure. Grounding, visibly working on an on-topic query.
- **No hallucinated "closest" source.** `"bake sourdough"` clears neither arm, so the gate
  returns empty and no model runs — the same result under `extractive` and `gemini`.

```
python -m rag_support_agent.generation.ask --query "How do I rotate an API key?"   # keyless
LLM_PROVIDER=gemini python -m rag_support_agent.generation.ask \
  --query "What is the per-call overage rate in USD?"                              # real, abstains
```

**Trade-off / what breaks when it fails.** `extractive` is grounded but not fluent — it
echoes, it can't rephrase or stitch passages together. `gemini`'s grounding is *prompt-*
enforced, not proven: it can still paraphrase-drift or cite loosely. This layer kills the
*easy* hallucinations (no source; on-topic-but-unanswerable); it does **not** certify
faithfulness — that becomes a measured number in M5, and confidence-calibrated abstention is
M4. `Answer.confidence` here is a deliberate placeholder (the top RRF score) that **no
decision reads yet**; M3 abstention keys only off retrieval and grounding.

### Confidence signal: how it's computed and calibrated

**Decision.** Score each answer with a confidence in `[0,1]` and abstain below a threshold —
a *third* abstention path on top of M3's two. Confidence is
`retrieval_spread × grounding_factor`. The backbone, `retrieval_spread`, is how far the top
hit stands above the field on **dense cosine similarity**:
`(d_top − mean(d_of_the_rest)) / d_top`. A clear winner scores high; a flat field of
near-ties scores ~0. When it's below `confidence_abstain_threshold`, the agent abstains and
**points at the closest source** ("I don't have a confident answer — the closest source is
X") — deliberately different from the structural refusal's "nothing here."

**Why spread, and not the three tempting alternatives.**
- **Not the fused RRF score.** It's rank-based and magnitude-blind: for the obvious query
  `"How do I rotate an API key?"` the top RRF is **0.0328** and the runner-up **0.0318** —
  a dead heat that says nothing about how clear the winner is. RRF is the right *ranking*
  signal and the wrong *confidence* signal.
- **Not an absolute cosine floor** (and *not* the M2 gate's 0.15). Absolute cosine doesn't
  transfer across embedders — M2 measured Gemini packing even out-of-scope hits at ~0.43.
  A floor tuned on one embedder silently mis-fires on another. Spread is *relative*, so it
  transfers: a real winner separates from the pack regardless of the absolute band.
- **Not BM25 folded into confidence.** BM25 is unbounded and corpus-scale-dependent;
  including it *destroyed* the separation in testing (it made ambiguous queries look
  confident on incidental keyword overlap). So the two signals get clean, separate jobs:
  **BM25 decides *in-scope* (the M2 gate); dense spread decides *confident* (here).**

**The grounding factor.** Spread answers "is there a clear winner"; grounding answers "is the
drafted answer actually supported by it." For both shipped generators this is **1.0** by
construction — `ExtractiveGenerator` echoes retrieved text verbatim, and `GeminiGenerator` is
pinned to the numbered context (an unsupported answer becomes the M3 Layer-2 sentinel, caught
before we get here). The optional enhancement — an **LLM self-eval** scoring entailment of the
draft — plugs in *here* as a `<1.0` factor for the gemini path. It stays **off by default** so
the pipeline needs no API key; turning "grounded?" into a measured number is M5.

**Challenge — the signal's quality is embedder-bound, and I measured it honestly.** Under the
keyless `hash` embedder the dense arm is *lexical* (≈ BM25, as the M2 write-up shows), so
spread is muted and the confident/ambiguous bands **overlap** — no threshold cleanly separates
on this 28-chunk corpus (flagship `rotate` spreads 0.34 while ambiguous `"tell me about
limits"` spreads ~0.53). The signal only comes alive with a **semantic** embedder, exactly
where M2 predicted real abstention would live. So the keyless path *computes* confidence and
*can* abstain, but the clean separation is demonstrated under Gemini:

| Provider (embedder) | Query | Verdict | Confidence | Which layer |
|---|---|---|---|---|
| `hash` (keyless) | `How do I rotate an API key?` | answered | **0.340** | — |
| `hash` (keyless) | `how do I bake sourdough bread` | abstained | 0.000 | 1 · structural (gate empty) |
| `gemini` | `How do I rotate an API key?` | answered | **0.158** | — |
| `gemini` | `How do I verify a webhook signature?` | answered | **0.223** | — |
| `gemini` | `What is the per-call overage rate in USD?` | abstained | 0.000 | 2 · sentinel (on-topic, unanswerable) |
| `gemini` | `How do I get started?` | **abstained** | **0.095** | **3 · confidence (this milestone)** |
| `gemini` | `how do I bake sourdough bread` | **abstained** | **0.032** | **3 · confidence** |

Under Gemini the spread separates cleanly across the wider probe set: **every clear query
lands in 0.14–0.28, every ambiguous or out-of-scope one at ≤ 0.095** — so the
`confidence_abstain_threshold = 0.12` cut splits them. Two results are worth calling out:

- **The new capability.** `"How do I get started?"` clears the gate and the generator *does*
  produce an answer (extractive can't refuse) — yet M4 withholds it, because the retrieval is
  an ambiguous scatter with no clear winner (spread 0.095), and points at the closest source
  instead. That's the third abstention doing something the M3 layers can't.
- **Spread is the out-of-scope defense a semantic embedder needs.** `"bake sourdough bread"`
  abstains *structurally* under `hash` (the gate returns empty) — but under Gemini the 0.15
  gate floor doesn't transfer, so the query **leaks past the gate** with hits at cosine ~0.44.
  Nothing separates from that flat field (spread 0.032), so the confidence layer catches the
  out-of-scope query the gate no longer can. M2's gate and M4's confidence cover each other's
  blind spots.

**Trade-off / what breaks.** The `0.12` threshold is *one* defensible cut taken from the
measured CLEAR/ambiguous boundary — precise calibration against the labeled set (abstention
precision/recall) is **M5**, not guesswork here. A **lone** gated hit scores spread 0 (there's
no field to stand out from) and is abstained conservatively — a genuinely unique match can be
refused, a deliberate trade to never claim false confidence from a single incidental hit. And
the honest limit above stands: under a purely lexical embedder the signal is too weak to
trust — confidence-calibrated abstention wants a semantic embedder underneath it.

### Knowledge decay: modeling staleness without ground truth — _tbd (M6)_
### Self-hosted pgvector vs a managed vector DB (cost/control trade-off) — _tbd_

## License

GPLv3 — see [LICENSE](LICENSE). Copyleft on purpose: this is a public reference
implementation, and I want derivatives to stay open.
