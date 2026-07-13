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
- ✅ **Write-up:** "Why hybrid retrieval over pure vector" — two reproducible keyless
  measurements (BM25 recovers a source vector's gate dropped; gate abstains on out-of-scope).
- ⏳ Semantic/paraphrase half of the write-up needs a real embedder (no OpenAI key in this
  env). Reproduce recipe is in the README; drop in numbers when a key is available.
- *Demo:* `retrieval.search --query "401 Unauthorized error" --show-arms`.

## M3 — Generation + citations
- Answer synthesis grounded strictly in retrieved context, with inline **citations**.
- Refuse to answer from parametric memory (grounding-only).
- *Demo:* question → cited answer.

## M4 — Confidence + abstention (the trust layer)
- Compute a confidence signal (retrieval score spread + grounding check + optional self-eval).
- Below threshold → **abstain**: "I don't have a confident answer — here's the closest source / who to ask."
- **Write-up:** how the confidence signal is computed and calibrated.
- *Demo:* an out-of-scope question that correctly gets "I don't know".

## M5 — Evaluation harness (the differentiator)
- Labeled Q/A set in `evaluation/datasets/` (gold question → gold source → gold answer).
- Metrics: Recall@k, answer faithfulness (grounded?), abstention precision, latency, cost.
- One command → the numbers in the README table.
- **Write-up:** how you measure faithfulness without a human in the loop.
- *Demo:* `eval.run` prints the table.

## M6 — Knowledge freshness / decay
- Score staleness from source age + update signals; surface a "possibly stale" flag on answers.
- **Write-up:** modeling decay without ground truth.
- *Demo:* age a source → answer gets flagged.

## M7 — Blind-spot detection + observability
- Persist every low-confidence / no-source query → **knowledge-gap report** ("top 10 things users ask that we can't answer").
- Per-request cost/latency/token logging.
- **Write-up:** closing the loop — the bot tells you what docs to write next.
- *Demo:* the gap report.

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
