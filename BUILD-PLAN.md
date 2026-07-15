# Build plan

Ordered so that **every milestone is demoable** and the repo is presentable even if you stop early.
Each milestone maps to a design-decision write-up in the README (that's your interview material).

## Guardrails (non-negotiable)
- **Clean-room.** Not one line of employer code, not one row of employer data. Generic domain only.
- **Neutral data.** Default: public docs of an open-source project OR synthetic SaaS docs generated here.
- **Every technique gets a README write-up:** decision → why → challenge → how you solved it → what you measured.

---

## M0 — Skeleton (this scaffold) ✅
Repo structure, README-pitch, Docker Compose (pgvector), pyproject, stubs.

## M1 — Ingestion + knowledge units ✅
- ✅ Loader for the sample docs (Markdown), capturing source mtime for freshness.
- ✅ Heading-aware chunking with size+overlap windowing (unit-tested, 4 tests green).
- ✅ **Knowledge unit** = chunk + metadata (source, section, chunk_index, content_hash, source_updated_at).
- ✅ Pluggable embedders: `hash` (keyless dev/test) + `openai`; idempotent upsert into pgvector.
- ✅ Runs end-to-end keyless: `python -m rag_support_agent.ingestion.run --source data/sample_docs --dry-run`
  → 5 docs → 28 chunks → embedded.
- ✅ **Write-up (README design-decisions):** chunking strategy & why.
- *Demo:* `docker compose up -d` then ingest (no `--dry-run`) → rows in Postgres.

## M2 — Retrieval (the core) ✅
- ✅ Dense: pgvector cosine top-k (`<=>`, matches the HNSW `vector_cosine_ops` index);
  query embedded with the same provider as ingest; heading path folded into the embedding.
- ✅ Sparse: **BM25** (`rank-bm25`) with an error-code-preserving tokenizer + stopword removal.
- ✅ **Fuse** with reciprocal rank fusion (rank-based → no score normalization / weight tuning).
- ✅ Relevance gate: keep only if dense clears a cosine floor OR sparse has a keyword hit;
  out-of-scope → empty (feeds M4 abstention).
- ✅ Demo CLI `retrieval.search --query ... --show-arms` (hybrid + per-arm breakdown).
- ✅ Unit tests for fusion / tokenizer / gate (DB-free): `tests/test_fusion.py`.
- ✅ **Write-up:** "Why hybrid retrieval over pure vector" — reproducible keyless
  measurements (BM25 recovers a source vector's gate dropped; gate abstains on out-of-scope)
  + the semantic half measured with real embeddings.
- ✅ Semantic/paraphrase half: `GeminiEmbedder` (`gemini-embedding-001`, 1536-d, L2-normalized).
  Measured — dense finds `Revoking a key` #1 from a keyword-free paraphrase that BM25 misses;
  and that the 0.15 cosine floor doesn't transfer to Gemini's compressed score band (→ M5 calibration).
- *Demo:* `retrieval.search --query "401 Unauthorized error" --show-arms`.

## M3 — Generation + citations ✅
- ✅ Answer synthesis grounded strictly in retrieved context, with inline **citations**.
- ✅ Refuse to answer from parametric memory (grounding-only, two layers: structural gate + sentinel).
- ✅ Pluggable generator (`extractive` keyless / `gemini`); citations carry the true marker index.
- *Demo:* question → cited answer.

## M4 — Confidence + abstention (the trust layer) ✅
- ✅ Confidence signal `= retrieval_spread × grounding_factor` in `[0,1]`. Backbone = **dense-cosine
  score spread** (top hit vs. the field) — relative, so it transfers across embedders; *not* the
  RRF score (magnitude-blind) and *not* the M2 gate's absolute cosine floor (doesn't transfer).
- ✅ Grounding factor = 1.0 for both shipped generators (extractive verbatim / gemini prompt-pinned);
  optional LLM self-eval is a pluggable `<1.0` factor for the gemini path, off by default (keyless).
- ✅ **Third abstention** (distinct from M3's two): retrieved + answered but ambiguous (spread below
  `confidence_abstain_threshold`) → abstain, pointing at the **closest source**.
- ✅ **Write-up:** how the confidence signal is computed + measured (keyless-coarse vs. Gemini-separated).
- ✅ *Demo:* a query that clears the gate and the generator answers, but M4 abstains on low spread
  (`"How do I get started?"`); confirmed under Gemini where the signal separates cleanly.
- Threshold *calibration* against the labeled set (abstention precision/recall) is **M5**.

## M5 — Evaluation harness (the differentiator) ✅
- ✅ Labeled Q/A set in `evaluation/datasets/` — 23 cases across 4 classes (answerable,
  ambiguous, out-of-scope, on-topic-but-unanswerable), gold source + gold answer +
  `expected_abstain`. Clean-room: only the synthetic 5-doc / 28-chunk domain.
- ✅ Metrics: Recall@k, answer faithfulness (LLM-judge entailment), abstention
  **precision *and* recall**, p50/p95 latency, approximate cost/1k. Pure, unit-tested
  (`tests/test_eval.py`); an instrumented runner pinned to `build_answer`'s verdict.
- ✅ One command → the README table (`python -m rag_support_agent.eval.run --dataset ...`),
  with `--calibrate` for the threshold sweep. Keyless subset vs. Gemini-gated full suite,
  same auto-detect pattern as M2/M4.
- ✅ **Threshold calibration** (the fil rouge from M4): the sweep *validates* the provisional
  `0.12` — F1 peaks (77.8%) across τ∈[0.10, 0.15]; under `hash` the curve is flat (signal
  muted), exactly M4's honest limit now shown as a measured curve.
- ✅ **Write-up:** measuring faithfulness without a human (claim-level LLM-judge entailment;
  the judge never sees the question — faithfulness ≠ correctness) + the calibration curve.
- *Demo:* `eval.run` prints the table; `--calibrate` prints the precision/recall curve.

## M6 — Knowledge freshness / decay ✅
- ✅ Score decay *risk* from source age (`source_updated_at`) — two signals: absolute
  half-life (`0.5 ** (age/half_life)`) + relative age-outlier vs the retrieved field
  (transfers across absolute bands, cf. M4; silent on a uniform / fresh-clone corpus).
- ✅ Surface a "possibly stale" flag on answers: `knowledge/freshness.py` (pure, keyless,
  injectable `now`), wired in `build_answer` → `Answer.stale_sources` (scoped to the cited
  sources), shown in the `ask` CLI. Unit-tested (`tests/test_freshness.py`, 17 cases).
- ✅ **Write-up:** "modeling decay without ground truth" — mtime is a checkout artifact, age
  is a risk proxy (not a verdict), and production swaps in a real content-change timestamp.
- ✅ *Demo:* `touch -d '2 years ago' data/sample_docs/api-keys.md` + re-ingest → freshness
  0.977 → 0.060, `api-keys.md` flagged on the answer; the intact corpus flags nothing.

## M7 — Blind-spot detection + observability ✅
- ✅ Persist **every** served query to an append-only `query_events` log (`observability/
  blindspot.py`, same `get_conn`/`init` pattern as `knowledge/db.py`); the `abstained` subset
  (M3 no-source ∪ sentinel ∪ M4 low-confidence) becomes the **knowledge-gap report** — "top
  things users ask that we can't answer." Append-only *on purpose*: a repeated question is
  volume, not a duplicate to collapse.
- ✅ Per-request **cost / latency / tokens**: `Answer.cost_usd` priced cross-provider in the
  thin `answer_question` layer (reusing M5 `eval/cost.py`; `build_answer` stays pure), $0 keyless;
  end-to-end latency + token counts on each event. Logging is opt-in + fail-soft (eval/tests
  bypass it; a telemetry failure never fails an answer).
- ✅ **Keyless-coarse vs. gated** clustering, the recurring pattern: `cluster_lexical`
  (connected components over the *same* BM25 tokenizer, keyless) and `cluster_semantic`
  (embedding cosine, gated — measurable now since Gemini *embeddings* dodge the generate cap).
- ✅ **Write-up:** "closing the loop" — measured the report + observability panel over the
  23-case log (8 abstained/23, ~30/~50 ms, $0), and the lexical-vs-semantic divergence (a
  disjoint-vocab paraphrase at cosine 0.685 vs unrelated ≤0.502; lexical splits it, semantic
  merges it to one `2×` theme). Honest limits: single-linkage chaining, embedder-specific
  threshold, PII of logging real queries.
- ✅ Unit-tested (`tests/test_blindspot.py`, pure/DB-free): cost pricing, tokenizer, lexical +
  semantic clustering, and that `build_answer` stays cost/log-free.
- ✅ *Demo:* `observability.replay` seeds the log → `observability.gap_report` prints the
  themes + cost/latency panel; `--mode semantic` clusters paraphrases by meaning.

## M8 — Thin chat UI + polish
- Minimal chat interface (streaming, citations, confidence badge, stale flag).
- Loom demo video (2–3 min) for outreach + interviews.
- Screenshot/GIF in README.

---

## Definition of "portfolio-ready"
- `docker compose up` + `uv sync` + one ingest command → it runs for a stranger.
- `eval.run` produces real numbers.
- README design-decision write-ups are filled in.
- One 2–3 min demo video linked.

Ship M1–M5 and it already beats 95% of RAG repos. M6–M8 are the senior signals.
