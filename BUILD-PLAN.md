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

## M2 — Retrieval (the core)
- Embeddings (pluggable provider) → pgvector; cosine top-k.
- Add **BM25 keyword** search; **fuse** (reciprocal rank fusion) → hybrid.
- Relevance gate: drop weak matches before generation.
- **Write-up:** why hybrid beats pure vector (error codes, exact API names).
- *Demo:* query → ranked sources with scores.

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
