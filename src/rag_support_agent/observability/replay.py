"""Seed the query-event log by replaying a labeled dataset through the pipeline (M7 demo).

    python -m rag_support_agent.observability.replay \
        --dataset evaluation/datasets/support_qa.jsonl --reset

Runs each question through ``answer_question`` with logging on, so the blind-spot log fills
with realistic traffic — the answerable cases plus the ambiguous / out-of-scope ones that
abstain (which are exactly what the gap report mines). Clean-room + privacy: the "traffic" is
the synthetic 23-case eval set, never a real user query. Requires an ingested database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_support_agent.config import get_settings
from rag_support_agent.generation.answer import answer_question
from rag_support_agent.knowledge.db import get_conn
from rag_support_agent.knowledge.models import AnswerVerdict
from rag_support_agent.observability.blindspot import init_events_schema


def _reset(settings) -> None:
    """Truncate the log so a repeated demo run starts clean instead of accumulating."""
    conn = get_conn(settings)
    try:
        init_events_schema(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE query_events RESTART IDENTITY;")
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a dataset to seed the query-event log.")
    parser.add_argument("--dataset", required=True, help="JSONL with a 'question' field per line.")
    parser.add_argument("--reset", action="store_true", help="Truncate the log before replaying.")
    args = parser.parse_args()

    settings = get_settings()
    if args.reset:
        _reset(settings)

    lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line)["question"] for line in lines if line.strip()]

    answered = abstained = 0
    for q in questions:
        ans = answer_question(q, settings=settings, record_event=True)
        if ans.verdict is AnswerVerdict.ABSTAINED:
            abstained += 1
        else:
            answered += 1

    print(
        f"replayed {len(questions)} queries -> {answered} answered, {abstained} abstained "
        f"(provider={settings.llm_provider} / embed={settings.embedding_provider}); "
        "logged to query_events."
    )


if __name__ == "__main__":
    main()
