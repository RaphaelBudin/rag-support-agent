"""Evaluation entrypoint:  python -m rag_support_agent.eval.run --dataset evaluation/datasets/support_qa.jsonl

Runs the labeled Q/A set through the agent and prints the metrics that go in the
README table. This is the part that turns "seems good" into numbers.

Metrics (M5):
  - Recall@k        : is the gold source in the top-k retrieved?
  - Faithfulness    : is the answer entailed by retrieved context? (no hallucination)
  - Abstention prec.: when it abstains, was abstaining correct?
  - Latency p95 / cost per 1k queries

TODO(M5): implement the harness.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the agent against a labeled set.")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL eval set.")
    args = parser.parse_args()
    print(f"[eval] TODO: run agent over {args.dataset!r} and print metrics table")


if __name__ == "__main__":
    main()
