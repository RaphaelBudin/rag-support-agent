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

# 4. Run the API + UI
python -m rag_support_agent.api.server
# open http://localhost:8000

# 5. Run the eval harness
python -m rag_support_agent.eval.run --dataset evaluation/datasets/support_qa.jsonl
```

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

Short write-ups of the non-obvious calls — this is where the engineering lives:

- **Why hybrid retrieval over pure vector** — _tbd_
- **Chunking strategy & why** — _tbd_
- **Confidence signal: how it's computed and calibrated** — _tbd_
- **Knowledge decay: modeling staleness without ground truth** — _tbd_
- **Self-hosted pgvector vs a managed vector DB (cost/control trade-off)** — _tbd_

## License

MIT — see [LICENSE](LICENSE).
